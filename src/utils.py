import numpy as np
import tensorflow_datasets as tfds


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
            cls_split[max_idx] -= sum(cls_split)-cls_ex_num
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
