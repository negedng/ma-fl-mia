import flwr as fl
from flwr.common.logger import log
from flwr.common import ndarrays_to_parameters
from logging import ERROR, INFO
import shutil
from datetime import datetime
import os
import numpy as np
import json
import copy

from src.flower_client import FlowerClient
from src.flower_strategy import SaveAndLogStrategy
from src import utils, models, data_preparation, attacks, metrics
from exp import setups


global conf 
global X_split
global Y_split 
global X_val
global Y_val

conf = utils.load_config()
os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]


def client_fn(cid: str) -> fl.client.Client:
    """Prepare flower client from ID (following flower documentation)"""
    client = FlowerClient(int(cid), conf)
    client.load_data(X_split[int(cid)], Y_split[int(cid)], X_val, Y_val)
    client.init_model()
    return client


def train(conf, train_ds=None):
    global X_split
    global Y_split
    conf['model_id'] = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(conf['paths']['models'], conf['model_id']), mode=0o777)
    shutil.copy(conf['paths']['code'],
                os.path.join(conf['paths']['models'], conf['model_id'], "train.py"))
    with open(os.path.join(conf['paths']['models'], conf['model_id'], "config.json"), "w") as f:
        json.dump(conf, f, indent=4)
    
    if train_ds is None:
        train_ds, _, _ = tfds.load('cifar10', split=['train[5%:]','train[:5%]','test'], as_supervised=True)
        
    X_train, Y_train = utils.get_np_from_tfds(train_ds)
    conf['len_total_data'] = len(X_train)
    X_split, Y_split = data_preparation.split_data(X_train, Y_train, conf['num_clients'], split_mode=conf['split_mode'],
                                                   mode="clients", seed=conf['seed'], dirichlet_alpha=conf['dirichlet_alpha'])

    initial_model = models.get_model(training_phase=True, unit_size=conf['unit_size'], conf=conf)
    if conf["continue_from"] is not None:
        print(f'load weights from checkpoint {conf["continue_from"]}')
        initial_model.load_weights(conf["continue_from"])
    initial_model.compile(optimizer=models.get_optimizer(),
                  loss=models.get_loss(),
                  metrics=["accuracy"])

    # Create FedAvg strategy
    strategy = SaveAndLogStrategy(
        conf=conf,
        initial_parameters = ndarrays_to_parameters(initial_model.get_weights()), # avoid smaller models as init
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
    model = models.get_model(unit_size=conf['unit_size'], conf=conf)
    model.load_weights(os.path.join(conf['paths']['models'], conf['model_id'], "saved_model"))
    model.compile(optimizer=models.get_optimizer(),
                  loss=models.get_loss(),
                  metrics=["accuracy"])
    return model, conf

                                                  
        
if __name__ == "__main__":
    import argparse
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Select experiment from exp.setup, for single run change config.json')
    parser.add_argument('--exp', type=str, help='See full list in exp/setup.py', default='default')
    args = parser.parse_args()
    conf_changes = setups.get_experiment(args.exp)
    

    import tensorflow as tf 
    import tensorflow_datasets as tfds
    if conf['val_split']:
        train_ds, val_ds, test_ds = tfds.load('cifar10', split=['train[5%:]','train[:5%]','test'], as_supervised=True)
        X_val, Y_val = utils.get_np_from_tfds(val_ds)        
    else:
        val_ds = None
        train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
        X_val, Y_val = utils.get_np_from_tfds(test_ds)
    f_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    
    with open(os.path.join(os.path.dirname(conf['paths']['code']),f'dump/{f_name}.json'), 'w') as f:
        f.write("[\n") 
        
    orig_conf = copy.deepcopy(conf)
    for cc in conf_changes:
        for k,v in cc.items():
            conf = copy.deepcopy(orig_conf)
            conf[k] = v
        print(conf)                
        model, model_conf = train(conf, train_ds)
        
        print("Training completed, model evaluation")
        # Evaluate
        results = metrics.evaluate(model_conf, model, train_ds, val_ds, test_ds)
        print(results)
        with open(os.path.join(model_conf['paths']['models'], model_conf['model_id'], "tests.json"), 'w') as f:
            f.write(json.dumps(results))

        with open(os.path.join(os.path.dirname(conf['paths']['code']),f'dump/{f_name}.json'), 'a') as f:
            f.write("  "+json.dumps(results)+",\n")
        
        # Per client eval
        results = metrics.evaluate_per_client(model_conf, model, X_split, Y_split, train_ds, val_ds, test_ds)
        with open(os.path.join(model_conf['paths']['models'], model_conf['model_id'], "client_results.json"), 'w') as f:
            f.write(json.dumps(results))

# endfor                      
    
    with open(os.path.join(os.path.dirname(conf['paths']['code']),f'dump/{f_name}.json'), 'a') as f:
        f.write("\n]")    
    
