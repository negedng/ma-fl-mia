import numpy as np
import tensorflow_datasets as tfds
import os

from src.utils import get_np_from_tfds


def dirichlet_split(num_classes, num_clients, dirichlet_alpha=1.0, mode="classes", seed=None):
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
    if np.isscalar(dirichlet_alpha):
        dirichlet_alpha = np.repeat(dirichlet_alpha, a)
    split_norm = np.random.default_rng(seed).dirichlet(dirichlet_alpha, b)
    return split_norm


def split_data(X, Y, num_clients, split=None, split_mode='dirichlet', *args, **kwargs):
    """Split data in X,Y between 'num_clients' number of clients"""
    assert len(X)==len(Y)
    classes = np.unique(Y)
    num_classes = len(classes)
    
    if split is None:
        if split_mode=='dirichlet':
            split = dirichlet_split(num_classes, num_clients, *args, **kwargs)
        elif split_mode=='binary':
            if num_clients == 20:
                split = [4/50]*10 + [1/50]*10
            elif num_clients == 10:
                split = [9/50]*5 + [1/50]*5
            split = [split]*10
            split = np.array(split)
        elif split_mode=='homogen':
            split = [1/num_clients]*num_clients
            split = [split]*10
            split = np.array(split)
        else:
            ValueError(f'Split mode not recognized {split_mode}')
    X_split = None
    Y_split = None
    idx_split = None
    
    for i, cls in enumerate(classes):
        idx_cls = np.where(Y==cls)[0]
        cls_num_example = len(idx_cls)
        cls_split = np.rint(split[i]*cls_num_example)
        
        # if rounding error remove it from most populus one
        if sum(cls_split)>cls_num_example:
            max_val = np.max(cls_split)
            max_idx = np.where(cls_split == max_val)[0][0]  
            cls_split[max_idx] -= sum(cls_split)-cls_num_example
        cls_split = cls_split.astype(int)
        idx_cls_split = np.split(idx_cls, np.cumsum(cls_split)[:-1])   
        if idx_split is None:
            idx_split = idx_cls_split
            
        else:
            for i in range(len(idx_cls_split)):

                idx_split[i] = np.concatenate([idx_split[i],idx_cls_split[i]], axis=0)
    
    for i in range(len(idx_split)):
        idx_split[i] = np.sort(idx_split[i])
        
    X_split = np.array([X[idx] for idx in idx_split])
    Y_split = np.array([Y[idx] for idx in idx_split])
    return X_split, Y_split


def get_mia_datasets_client_balanced(X_split, Y_split, X_test, Y_test, n_attacker_knowledge=100, n_attack_sample=5000, seed=None):
    """Get attacker knowledge from train split-wise balanced"""
    no_clients = len(Y_split)
    
    n_attacker_knowledge_cls = n_attacker_knowledge // no_clients
    n_attack_sample_cls = n_attack_sample // no_clients
    
    for i in range(len(Y_split)):
        p = np.random.RandomState(seed=seed).permutation(len(Y_split[i]))
        idx_train_attacker = p[:n_attacker_knowledge_cls]
        idx_test_train = p[-n_attack_sample_cls:]
        if i==0:
            x_train_attacker = X_split[i][idx_train_attacker]
            y_train_attacker = Y_split[i][idx_train_attacker]
            x_test_train = X_split[i][idx_test_train]
            y_test_train = Y_split[i][idx_test_train]
        else:
            x_train_attacker = np.concatenate([x_train_attacker, X_split[i][idx_train_attacker]],axis=0)
            y_train_attacker = np.concatenate([y_train_attacker, Y_split[i][idx_train_attacker]],axis=0)  
            x_test_train = np.concatenate([x_test_train, X_split[i][idx_test_train]],axis=0)
            y_test_train = np.concatenate([y_test_train, Y_split[i][idx_test_train]],axis=0)
    p = np.random.RandomState(seed=seed).permutation(len(Y_test))
    idx_test_attacker = p[:n_attacker_knowledge]
    idx_test_test = p[-n_attack_sample_cls:]     
    x_test_attacker = X_test[idx_test_attacker]
    y_test_attacker = Y_test[idx_test_attacker]
    x_test_test = X_test[idx_test_test]
    y_test_test = Y_test[idx_test_test]
    
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
    
def get_mia_datasets(train_ds, test_ds, n_attacker_knowledge=100, n_attack_sample=5000, seed=None):
    """Get attacker training data knowledge 
    and sample from train and test set for attack evaluation."""

    train_ds_attacker = train_ds.shuffle(50000, seed=seed).take(n_attacker_knowledge)
    test_ds_attacker = test_ds.shuffle(10000, seed=seed).take(n_attacker_knowledge)

    test_from_test_ds = test_ds.shuffle(10000, seed=seed).take(n_attack_sample)
    test_from_train_ds = train_ds.shuffle(50000, seed=seed).take(n_attack_sample)

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

