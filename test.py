import shutil
from datetime import datetime
import os
import numpy as np
import json
import tensorflow as tf 
import tensorflow_datasets as tfds

from src import utils, models, data_preparation, attacks, metrics

def test(model_path, overwrite=False):
    
    config_path = os.path.join(model_path, 'config.json')
    conf = utils.load_config(config_path=config_path)
    conf['paths']['models']
    print(conf)
    train_ds, val_ds, test_ds = data_preparation.load_and_preprocess(conf=conf)
    X_train, Y_train = utils.get_np_from_tfds(train_ds)
    conf['len_total_data'] = len(X_train)        
    if 'split_mode' not in conf.keys():
        split_mode='dirichlet'
    else:
        split_mode=conf['split_mode']
    X_split, Y_split = data_preparation.split_data(X_train, Y_train, conf['num_clients'], split_mode=split_mode,
                                                   mode="clients", seed=conf['seed'], dirichlet_alpha=conf['dirichlet_alpha'])
    
    if os.path.exists(os.path.join(model_path,'saved_model')):
        weights_path = os.path.join(model_path,'saved_model')
    else:
        weights_path = os.path.join(model_path,'saved_model_best')        
    model = models.get_model(unit_size=conf['unit_size'], conf=conf)
    model.load_weights(os.path.join(weights_path)).expect_partial()
    model.compile(optimizer=models.get_optimizer(),
                  loss=models.get_loss(),
                  metrics=["accuracy"])
    
    print(f'Testing model {model_path}')
    # Evaluate
    # Per client attack
    if overwrite or not os.path.exists(os.path.join(model_path, "client_results.json")):
        results = metrics.attack_on_clients(conf, X_split, Y_split, train_ds, val_ds, test_ds)
        print(results)
        with open(os.path.join(model_path, "client_results.json"), 'w') as f:
            f.write(json.dumps(results))    
    avg = results['average']        
    if overwrite or not os.path.exists(os.path.join(model_path, "tests.json")):
        results = metrics.evaluate(conf, model, train_ds, val_ds, test_ds)
        results['client_attacks'] = avg
        print(results)

        with open(os.path.join(model_path, "tests.json"), 'w') as f:
            f.write(json.dumps(results))
    



def test_all(model_path):
    
    config_path = os.path.join(model_path, 'config.json')
    conf = utils.load_config(env_path=None, config_path=config_path)
    print(conf)
    train_ds, val_ds, test_ds = data_preparation.load_and_preprocess(dataset_mode="cifar10", conf=conf)
    X_train, Y_train = utils.get_np_from_tfds(train_ds)
    conf['len_total_data'] = len(X_train)        
    if 'split_mode' not in conf.keys():
        split_mode='dirichlet'
    else:
        split_mode=conf['split_mode']
    X_split, Y_split = data_preparation.split_data(X_train, Y_train, conf['num_clients'], split_mode=split_mode,
                                                   mode="clients", seed=conf['seed'], dirichlet_alpha=conf['dirichlet_alpha'])
    
    subfolders = [ f.path for f in os.scandir(model_path) if f.is_dir() ]
    res = {}
    for weights_path in subfolders:
        if 'saved_model' not in weights_path:
            continue
        idx = os.path.basename(weights_path)     
        res[idx] = {} 
        model = models.get_model(unit_size=conf['unit_size'], conf=conf)
        model.load_weights(os.path.join(weights_path))
        model.compile(optimizer=models.get_optimizer(),
                      loss=models.get_loss(),
                      metrics=["accuracy"])
        
        print(f'Testing model {weights_path}')
        # Evaluate
        results = metrics.evaluate(conf, model, train_ds, val_ds, test_ds)

        res[idx]['global'] = results
        
        # Per client eval
        results = metrics.evaluate_per_client(conf, model, X_split, Y_split, train_ds, val_ds, test_ds)
        res[idx]['local'] = results
        print(results)
    with open(os.path.join(model_path, "all_model_results.json"), 'w') as f:
        f.write(json.dumps(res))  


if __name__ == "__main__":
    import argparse
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Select experiment from exp.setup, for single run change config.json')
    parser.add_argument('model_id', type=str, help='model id like: 20230424-221925')
    parser.add_argument('--root_path', type=str, help='root path for the model_id', default='/')
    parser.add_argument('--test_all', help='whether to run test_all or not', action='store_true')
    parser.add_argument('-o','--overwrite', help='whether to rewrite existing files', action='store_true')
    args = parser.parse_args()
    conf = utils.load_config()
    model_path = os.path.join(args.root_path,'data/models/ma-fl-mia/federated/',args.model_id)
    if args.test_all:
        test_all(model_path)
    else:
        test(model_path, args.overwrite)

