import flwr as fl
from flwr.common.logger import log
from logging import ERROR, INFO
import shutil
from datetime import datetime
import os
import numpy as np
import json

from src.flower_client import FlowerClient
from src.flower_strategy import SaveAndLogStrategy
from src import utils, models, data_preparation, attacks


global conf 
global X_split
global Y_split 
global X_test
global Y_test

conf = utils.load_config()
os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]


def client_fn(cid: str) -> fl.client.Client:
    """Prepare flower client from ID (following flower documentation)"""
    client = FlowerClient(int(cid), conf)
    client.load_data(X_split[int(cid)], Y_split[int(cid)], X_test, Y_test)
    client.init_model()
    return client


def train(conf, train_ds=None, test_ds=None):
    global X_split
    global Y_split
    conf['model_id'] = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(conf['paths']['models'], conf['model_id']), mode=0o777)
    shutil.copy(conf['paths']['code'],
                os.path.join(conf['paths']['models'], conf['model_id'], "train.py"))
    with open(os.path.join(conf['paths']['models'], conf['model_id'], "config.json"), "w") as f:
        json.dump(conf, f, indent=4)
    
    if train_ds is None:
        train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
        
    X_train, Y_train = utils.get_np_from_tfds(train_ds)
    X_split, Y_split = data_preparation.split_data(X_train, Y_train, conf['num_clients'],
                                                   mode="clients", seed=conf['seed'], alpha=conf['alpha'])

    # Create FedAvg strategy
    strategy = SaveAndLogStrategy(
        conf=conf,
        fraction_fit=1.0,  # Sample 10% of available clients for training
        fraction_evaluate=0.05,  # Sample 5% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
        # Wait until at least 75 clients are available
        # min_available_clients=1,
    )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=conf['num_clients'],
        config=fl.server.ServerConfig(num_rounds=conf['rounds']),
        strategy=strategy,
        ray_init_args = conf['ray_init_args'],
        client_resources = conf['client_resources']
    )
    model = tf.keras.models.load_model(os.path.join(conf['paths']['models'], conf['model_id'], "saved_model"),
                                       custom_objects={})
    model.compile(optimizer=models.get_optimizer(),
                  loss=models.get_loss(),
                  metrics=["accuracy"])
    return model


def evaluate(conf, model, train_ds=None, test_ds=None):
    if train_ds is None:
        train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
        
    r = data_preparation.get_mia_datasets(train_ds, test_ds,
                                          conf['n_attacker_knowledge'],
                                          conf['n_attack_sample'],
                                          conf['seed'])
    train_performance = model.evaluate(train_ds.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE))                                      
    test_performance = model.evaluate(test_ds.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE))
    mia_pred = attacks.attack(model,
                              r['attacker_knowledge'], r['mia_data'],
                              models.get_loss(),
                              attacks.get_af())
    advantage = attacks.calculate_advantage(r['mia_labels'], mia_pred)
    results = {
        'adv_std':advantage,
        'test_acc': test_performance[1],
        'train_acc': train_performance[1],
        'unit_size': conf['unit_size'],
        'alpha' : conf['alpha'],
        'model_id' : conf['model_id'],
        'params' : model.count_params(),
    }
    return results

if __name__ == "__main__":
    import tensorflow as tf 
    import tensorflow_datasets as tfds
    
    train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
    X_test, Y_test = utils.get_np_from_tfds(test_ds)
    
    f_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    for i, us in enumerate([5,10,15,20,30,40,50,60]):
        conf['unit_size'] = us
        model = train(conf, train_ds, test_ds)
        
        print("Training completed, model evaluation")
        # Evaluate
        results = evaluate(conf, model, train_ds, test_ds)
        print(results)
        if i==0:
            with open(os.path.join(os.path.dirname(conf['paths']['code']),f'dump/{f_name}.json'), 'w') as f:
                f.write("[\n")        
        with open(os.path.join(os.path.dirname(conf['paths']['code']),f'dump/{f_name}.json'), 'a') as f:
            f.write("  "+json.dumps(results)+",\n")
    
    with open(os.path.join(os.path.dirname(conf['paths']['code']),f'dump/{f_name}.json'), 'a') as f:
        f.write("\n]")    
    
