import numpy as np
import os
from tqdm import tqdm
import json


def get_np_from_tfds(ds):
    X = []
    Y = []
    for x,y in ds.as_numpy_iterator():
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


def load_config(env_path='env.json', config_path='config.json'):
    """Check for an env root file"""
    with open(config_path, "r") as f:
        config = json.load(f)
    if env_path is None:
        return config
    with open(env_path, "r") as f:
            env = json.load(f)
    config.update(env)
    for k,v in config['paths'].items():
        config['paths'][k] = str(os.path.join(config['root_path'], v))
    return config


def predict_losses(model, X, Y, loss_function, verbose=0.5):
    """Predict on model but returns with each individual loss"""
    losses = []
    
    p_verbose = 1.0 if verbose>0.75 else 0.0
    Y_pred = model.predict(X, verbose=p_verbose)
    
    iterator = zip(Y, Y_pred)
    if verbose>0.1:
        iterator = tqdm(iterator, total=len(Y))
    
    for y, y_pred in iterator:
        l = loss_function(y, y_pred)
        losses.append(l.numpy())
    return np.array(losses)


def select_n(X_train, Y_train, n):
    indexes = np.array([], dtype=int)
    classes = np.unique(Y_train)
    n_cls = n // len(classes)
    for c in classes:
        indexes = np.append(indexes, np.argwhere(Y_train==c)[:n_cls])
    return X_train[indexes], Y_train[indexes]
