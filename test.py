import shutil
from datetime import datetime
import os
import numpy as np
import json
import tensorflow as tf 
import tensorflow_datasets as tfds

from src import utils, models, data_preparation, attacks, metrics

def test(model_path):
    
    config_path = os.path.join(model_path, 'config.json')
    conf = utils.load_config(env_path=None, config_path=config_path)
    print(conf)
    if conf['val_split']:
        train_ds, val_ds, test_ds = tfds.load('cifar10', split=['train[5%:]','train[:5%]','test'], as_supervised=True)
    else:
        train_ds, test_ds = tfds.load('cifar10', split=['train','test'], as_supervised=True)
        val_ds = None
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
    model.load_weights(os.path.join(weights_path))
    model.compile(optimizer=models.get_optimizer(),
                  loss=models.get_loss(),
                  metrics=["accuracy"])
    
    print(f'Testing model {model_path}')
    # Evaluate
    if not os.path.exists(os.path.join(model_path, "tests.json")):
        results = metrics.evaluate(conf, model, train_ds, val_ds, test_ds)
        print(results)

        with open(os.path.join(model_path, "tests.json"), 'w') as f:
            f.write(json.dumps(results))
    
    # Per client eval
    if not os.path.exists(os.path.join(model_path, "client_results.json")):
        results = metrics.evaluate_per_client(conf, model, X_split, Y_split, train_ds, val_ds, test_ds)
        print(results)
        with open(os.path.join(model_path, "client_results.json"), 'w') as f:
            f.write(json.dumps(results))    
    

if __name__ == "__main__":
    model_path = '/data/models/ma-fl-mia/federated/20230419-114430'
    test(model_path)
