import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import os



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


def get_np_from_ds(ds):
    try:
        ds = ds.unbatch()
    except ValueError:
        # already unbatched
        pass
    X = []
    Y = []
    for x,y in ds.as_numpy_iterator():
        X.append(x)
        Y.append(y)
    return np.array(X), np.array(Y)
    
    
def get_mia_datasets(train_ds, test_ds, n_attacker_knowledge=100, n_attack_sample=5000, seed=None):
    """Get attacker training data knowledge 
    and sample from train and test set for attack evaluation."""

    train_ds_attacker = train_ds.shuffle(50000, seed=seed).take(n_attacker_knowledge)
    test_ds_attacker = test_ds.shuffle(10000, seed=seed).take(n_attacker_knowledge)

    test_from_test_ds = test_ds.shuffle(10000, seed=seed).take(n_attack_sample)
    test_from_train_ds = train_ds.shuffle(50000, seed=seed).take(n_attack_sample)

    x_train_attacker, y_train_attacker = get_np_from_ds(train_ds_attacker)
    x_test_attacker, y_test_attacker = get_np_from_ds(test_ds_attacker)
    x_test_test, y_test_test = get_np_from_ds(test_from_test_ds)
    x_test_train, y_test_train = get_np_from_ds(test_from_train_ds)
    
    real_n_attack_sample = min(len(y_test_test),len(y_test_train),n_attack_sample)
    x_test_test = x_test_test[:real_n_attack_sample]
    y_test_test = y_test_test[:real_n_attack_sample]
    x_test_train = x_test_train[:real_n_attack_sample]
    y_test_train = y_test_train[:real_n_attack_sample]

    x_mia_test = np.concatenate([x_test_train, x_test_test])
    y_mia_test = np.concatenate([y_test_train, y_test_test])
    mia_true = [1.0] * real_n_attack_sample + [0.0] * real_n_attack_sample
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


def preprocess(image, conf):
    input_shape=(32,32,3)
    if image.shape!=input_shape:
        image = tf.image.resize(image, input_shape[:2])
    if conf['data_normalize']:
        # Convert to float32 and scale pixel values to [0, 1]
        image = tf.cast(image, tf.float32) / 255.0
        # Subtract mean RGB values
    if conf['data_centralize']:
        mean_rgb = tf.constant([0.491, 0.482, 0.447])
        std_rgb = tf.constant([0.247, 0.243, 0.261])
        image = (image - mean_rgb) / std_rgb
    return image


def preprocess_ds(image, label, conf):
    image = preprocess(image, conf)
    return image, label


def preprocess_data(data, conf, shuffle=False):
    data = data.map(lambda x,y: preprocess_ds(x,y,conf)) 
    if shuffle:
        data = data.shuffle(5000)
    data = data.batch(conf['batch_size']).prefetch(tf.data.AUTOTUNE)
    return data


def load_data(dataset_mode="cifar10", input_shape=(32,32,3), val_split=True, norm=False, conf={}):
        
    if 'val_split' in conf.keys():
        val_split = conf['val_split']

    if dataset_mode == "cifar10":
        if val_split:
            train_ds, val_ds, test_ds = tfds.load('cifar10', split=['train[5%:]','train[:5%]','test'], as_supervised=True)
        else:
            train_ds, val_ds, test_ds = tfds.load('cifar10', split=['train','test','test'], as_supervised=True)
    
    return train_ds, val_ds, test_ds


def ds_from_numpy(data):
    return tf.data.Dataset.from_tensor_slices(data)