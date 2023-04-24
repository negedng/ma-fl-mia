from tqdm import tqdm
import numpy as np
import tensorflow as tf 
import tensorflow_datasets as tfds
from src import utils, models, data_preparation, attacks


def evaluate(conf, model, train_ds=None, val_ds=None, test_ds=None):
    if train_ds is None:
        train_ds, val_ds, test_ds = tfds.load('cifar10', split=['train[5%:]','train[:5%]','test'], as_supervised=True)
        
    r = data_preparation.get_mia_datasets(train_ds, test_ds,
                                          conf['n_attacker_knowledge'],
                                          conf['n_attack_sample'],
                                          conf['seed'])
    train_performance = model.evaluate(train_ds.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE)) 
    if val_ds is not None:
        val_performance = model.evaluate(val_ds.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE)) 
    else:
        val_performance = [None] * len(train_performance)                                     
    test_performance = model.evaluate(test_ds.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE))
    mia_preds = attacks.attack(model,
                              r['attacker_knowledge'], r['mia_data'],
                              models.get_loss())
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

def evaluate_per_client(conf, model, X_split, Y_split, train_ds=None, val_ds=None, test_ds=None):

    if train_ds is None:
        train_ds, val_ds, test_ds = tfds.load('cifar10', split=['train[5%:]','train[:5%]','test'], as_supervised=True)
    X_test, Y_test = utils.get_np_from_tfds(test_ds)            
    r = data_preparation.get_mia_datasets(train_ds, test_ds,
                                          conf['n_attacker_knowledge'],
                                          conf['n_attack_sample'],
                                          conf['seed'])
    results = []
    for X_client, Y_client in tqdm(zip(X_split, Y_split), total=len(X_split)):
        c_res = {}
        train_performance = model.evaluate(X_client, Y_client, verbose=0)
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
    
