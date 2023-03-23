import numpy as np
import os
from tqdm import tqdm


def get_np_from_tfds(ds):
    X = []
    Y = []
    for x,y in ds.as_numpy_iterator():
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)


def lookup_envroot(env_path='env', default_path=''):
    """Check for an env root file"""
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as file:
            return file.read().strip()
    return default_path


class Config:
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def predict_losses(model, X, Y, loss_function, verbose=0.5):
    """Predict on model but returns with each individual loss"""
    losses = []
    
    p_verbose = 1 if verbose>0.75 else 0
    Y_pred = model.predict(X, p_verbose)
    
    iterator = zip(Y, Y_pred)
    if verbose>0.1:
        iterator = tqdm(iterator, total=len(Y))
    
    for y, y_pred in iterator:
        l = loss_function(y, y_pred)
        losses.append(l.numpy())
    return losses

