from tqdm import tqdm
import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds
from src import utils, models, data_preparation, attacks
import os


def evaluate(conf, model, train_ds=None, val_ds=None, test_ds=None, verbose=1):
    """Attack on server"""
    if train_ds is None:
        train_ds, val_ds, test_ds = data_preparation.load_and_preprocess(conf=conf)
    
    train_ds = train_ds.map(lambda x,y: data_preparation.preprocess_ds(x,y,conf)) 
    test_ds = test_ds.map(lambda x,y: data_preparation.preprocess_ds(x,y,conf))    
    r = data_preparation.get_mia_datasets(train_ds, test_ds,
                                          conf['n_attacker_knowledge'],
                                          conf['n_attack_sample'],
                                          conf['seed'])
    train_performance = model.evaluate(train_ds.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=verbose) 
    if val_ds is not None:
        val_ds = val_ds.map(lambda x,y: data_preparation.preprocess_ds(x,y,conf)) 
        val_performance = model.evaluate(val_ds.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=verbose) 
    else:
        val_performance = [None] * len(train_performance)                                     
    test_performance = model.evaluate(test_ds.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE), verbose=verbose)
    mia_preds = attacks.attack(model,
                              r['attacker_knowledge'], r['mia_data'],
                              models.get_loss(),
                              verbose=verbose)
    results = {
        'test_acc': test_performance[1],
        'val_acc': val_performance[1],
        'train_acc': train_performance[1],
        'unit_size': conf['unit_size'],
        'alpha' : conf['dirichlet_alpha'],
        'model_id' : conf['model_id'],
        'params' : model.count_params(),
        "model_mode": conf["model_mode"],
        "scale_mode": conf["scale_mode"],
        "ma_mode": conf["ma_mode"]
    }
    for k, v in mia_preds.items():
        results[k] = attacks.calculate_advantage(r['mia_labels'], v)

    return results

def attack_on_clients(conf, X_split=None, Y_split=None, train_ds=None, val_ds=None, test_ds=None):
    """Attack on client"""
    if train_ds is None:
        train_ds, val_ds, test_ds = data_preparation.load_and_preprocess(conf=conf)
    if X_split is None:
        X_train, Y_train = utils.get_np_from_tfds(train_ds)
        conf['len_total_data'] = len(X_train)
        X_split, Y_split = data_preparation.split_data(X_train, Y_train, conf['num_clients'], split_mode=conf['split_mode'],
                                                       mode="clients", seed=conf['seed'], dirichlet_alpha=conf['dirichlet_alpha'])
    res = []
    for cid in tqdm(range(len(X_split))):
        model_root_path = conf['paths']['models']
        model_id = conf['model_id']
        last_epoch = str(conf['rounds'])
        epoch_id = "saved_model_post_"+last_epoch
        model_path = os.path.join(model_root_path, model_id, "clients", str(cid),epoch_id)
        local_unit_size = models.calculate_unit_size(cid, conf, len(X_split[cid]))
        conf["local_unit_size"] = local_unit_size
        model = models.get_model(unit_size=local_unit_size, conf=conf, keep_scaling=True) # maybe you need static_bn?
        model.load_weights(model_path).expect_partial()
        model.compile(optimizer=models.get_optimizer(learning_rate=conf['learning_rate']),
                      loss=models.get_loss(),
                      metrics=["sparse_categorical_accuracy"])
        train_c_ds = tf.data.Dataset.from_tensor_slices((X_split[cid],Y_split[cid]))
        r = evaluate(conf, model, train_c_ds, val_ds, test_ds, verbose=0)
        r["cid"] = cid
        r["local_unit_size"] = local_unit_size
        res.append(r)
    
    all_train_acc = [a['train_acc'] for a in res]
    avgs = {"avg_train_acc":np.mean(all_train_acc), "std_train_acc":np.std(all_train_acc)}
    all_test_acc = [a['test_acc'] for a in res]
    avgs["avg_test_acc"] = np.mean(all_test_acc)
    avgs["std_test_acc"] = np.std(all_test_acc)
    adv_list = []
    for k in res[0].keys():
        if "adv" in k:
            adv_list.append(k)
    for adv in adv_list:
        all_adv = [a[adv] for a in res]
        avgs["avg_"+adv] = np.mean(all_adv)
        avgs["std_"+adv] = np.std(all_adv)
        
    return {"client_results": res, "average":avgs}
                       

def evaluate_per_client(conf, model, X_split, Y_split, train_ds=None, val_ds=None, test_ds=None):
    """Server model attack with client data"""

    if train_ds is None:
        train_ds, val_ds, test_ds = data_preparation.load_and_preprocess(conf=conf)
    test_ds = test_ds.map(lambda x,y: data_preparation.preprocess_ds(x,y,conf))  
    X_test, Y_test = utils.get_np_from_tfds(test_ds)            
    r = data_preparation.get_mia_datasets_client_balanced(X_split, Y_split, X_test, Y_test,
                                          conf['n_attacker_knowledge'],
                                          conf['n_attack_sample'],
                                          conf['seed'])
    results = []
    for X_client, Y_client in tqdm(zip(X_split, Y_split), total=len(X_split)):
        c_res = {}
        train_ds = tf.data.Dataset.from_tensor_slices((X_client,Y_client))
        train_ds = train_ds.map(lambda x,y: data_preparation.preprocess_ds(x,y,conf))
        train_ds = train_ds.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE)
        
        train_performance = model.evaluate(train_ds, verbose=0)
        c_res['train_acc'] = train_performance[1]
        
        len_data = min(len(X_client), len(X_test))
        c_res['data_size'] = len(X_client)
        X_ctest = np.concatenate((X_client[:len_data], X_test[:len_data]))
        Y_ctest = np.concatenate((Y_client[:len_data], Y_test[:len_data]))
        mia_true = [1.0] * len_data + [0.0] * len_data
        mia_true = np.array(mia_true)
        mia_preds = attacks.attack(model,
                          r['attacker_knowledge'], (X_ctest, Y_ctest),
                          models.get_loss(),
                          verbose=0)
        for k, v in mia_preds.items():
            c_res[k] = attacks.calculate_advantage(mia_true, v)
        results.append(c_res)
    return results
    
