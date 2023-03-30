import numpy as np
import tensorflow_datasets as tfds
import os

from src.utils import get_np_from_tfds


def dirichlet_split(num_classes, num_clients, alpha=1.0, mode="classes", seed=None):
    """Dirichlet distribution of the data points, 
    with mode 'classes', 1.0 is distributed between num_classes class, 
    with 'clients' it is distributed between num_clients clients"""
    if mode=="classes":
        a = num_classes
        b = num_clients
    elif mode=="clients":
        a = num_clients
        b = num_classes
    else:
        raise ValueError(f'unrecognized mode {mode}')
    if np.isscalar(alpha):
        alpha = np.repeat(alpha, a)
    split_norm = np.random.default_rng(seed).dirichlet(alpha, b)
    return split_norm


def split_data(X, Y, num_clients, *args, **kwargs):
    """Split data in X,Y between 'num_clients' number of clients"""
    assert len(X)==len(Y)
    classes = np.unique(Y)
    num_classes = len(classes)
    
    split = dirichlet_split(num_classes, num_clients, *args, **kwargs)
    
    X_split = None
    Y_split = None
    
    for i, cls in enumerate(classes):
        X_cls = X[Y==cls]
        Y_cls = Y[Y==cls]
        
        cls_num_example = len(X_cls)
        cls_split = np.rint(split[i]*cls_num_example)
        
        # if rounding error remove it from most populus one
        if sum(cls_split)>cls_num_example:
            max_val = np.max(cls_split)
            max_idx = np.where(cls_split == max_val)[0][0]  
            cls_split[max_idx] -= sum(cls_split)-cls_num_example
        cls_split = cls_split.astype(int)

        X_cls_split = np.split(X_cls, np.cumsum(cls_split)[:-1])
        Y_cls_split = np.split(Y_cls, np.cumsum(cls_split)[:-1]) # these are just [cls]*cls_split   

        if X_split is None:
            X_split = X_cls_split
            Y_split = Y_cls_split
        else:
            for i in range(len(X_cls_split)):

                X_split[i] = np.concatenate([X_split[i],X_cls_split[i]], axis=0)
                Y_split[i] = np.concatenate([Y_split[i],Y_cls_split[i]], axis=0)
    
    return X_split, Y_split

    
def get_mia_datasets(train_ds, test_ds, n_attacker_knowledge=100, n_attack_sample=5000, seed=None):
    """Get attacker training data knowledge 
    and sample from train and test set for attack evaluation."""

    train_ds_attacker = train_ds.shuffle(10000, seed=seed).take(n_attacker_knowledge)
    test_ds_attacker = test_ds.shuffle(10000, seed=seed).take(n_attacker_knowledge)

    test_from_test_ds = test_ds.shuffle(10000, seed=seed).take(n_attack_sample)
    test_from_train_ds = train_ds.shuffle(10000, seed=seed).take(n_attack_sample)

    x_train_attacker, y_train_attacker = get_np_from_tfds(train_ds_attacker)
    x_test_attacker, y_test_attacker = get_np_from_tfds(test_ds_attacker)
    x_test_test, y_test_test = get_np_from_tfds(test_from_test_ds)
    x_test_train, y_test_train = get_np_from_tfds(test_from_train_ds)

    x_mia_test = np.concatenate([x_test_train, x_test_test])
    y_mia_test = np.concatenate([y_test_train, y_test_test])
    mia_true = [1.0] * n_attack_sample + [0.0] * n_attack_sample
    mia_true = np.array(mia_true)
    
    attacker_knowledge = {
        "in_train_data" : (x_train_attacker, y_train_attacker),
        "not_train_data" : (x_test_attacker, y_test_attacker),
    }
    
    ret = {
        "attacker_knowledge": attacker_knowledge,
        "mia_data" : (x_mia_test, y_mia_test),
        "mia_labels" : mia_true 
    }

    return ret

