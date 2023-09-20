import numpy as np
import os
import json
import logging
import re


def generalized_positional_notation(N, l):
    ret = [0] * len(l)
    quo = N
    for i in range(len(l) - 1, 0, -1):
        rem = quo % l[i]
        ret[i] = rem
        quo = quo // l[i]
    ret[0] = quo
    return ret


def load_config(env_path="env.json", config_path="config.json"):
    """Check for an env root file"""
    with open(config_path, "r") as f:
        config = json.load(f)
    if env_path is None:
        return config
    with open(env_path, "r") as f:
        env = json.load(f)
    for k, v in config["paths"].items():
        if "root_path" in config.keys():
            if v[: len(config["root_path"])] == config["root_path"]:
                v = v[len(config["root_path"]) :]
        config["paths"][k] = str(os.path.join(env["root_path"], v))
    config.update(env)
    return config


def select_n_from_each_class(X_train, Y_train, n):
    indexes = np.array([], dtype=int)
    classes = np.unique(Y_train)
    n_cls = n // len(classes)
    for c in classes:
        indexes = np.append(indexes, np.argwhere(Y_train == c)[:n_cls])
    return X_train[indexes], Y_train[indexes]


def get_random_permutation(cid, total_clients, seed):
    return np.random.RandomState(seed=seed).permutation(total_clients)[cid]


def get_random_permutation_for_all(cids, seed, total_clients=None, permutate="repeated"):
    if total_clients is None:
        total_clients = len(cids)
    if permutate=="repeated":
        rands = {
            cid:get_random_permutation(idx%total_clients, total_clients, seed) for idx, cid in enumerate(sorted(cids))
        }
    elif permutate=="incremental":
        rands = {
            cid:get_random_permutation(idx%total_clients, total_clients, seed)+seed*total_clients for idx, cid in enumerate(sorted(cids))
        }
    elif permutate=="static":
        rands = {
            cid:idx for idx, cid in enumerate(sorted(cids))
        }
    else:
        raise NotImplementedError(permutate)
    return rands


def calculate_unit_size(cid, conf, len_train_data):
    if conf["ma_mode"] == "heterofl":
        if conf["scale_mode"] == "standard":
            if len_train_data > 2500:
                unit_size = conf["unit_size"]
            elif len_train_data > 1250:
                unit_size = conf["unit_size"] // 2
            elif len_train_data > 750:
                unit_size = conf["unit_size"] // 4
            else:
                unit_size = conf["unit_size"] // 8
        elif conf["scale_mode"] == "basic":
            if len_train_data > 2500:
                unit_size = conf["unit_size"]
            else:
                unit_size = conf["unit_size"] // 2
        elif conf["scale_mode"] == "1250":
            if len_train_data > 1250:
                unit_size = conf["unit_size"]
            else:
                unit_size = conf["unit_size"] // 2
        elif conf["scale_mode"] == "no":
            unit_size = conf["unit_size"]
        elif type(conf["scale_mode"]) == float and conf["scale_mode"] <= 1.0:
            if cid%2==0:
                unit_size = conf["unit_size"]
            else:
                unit_size = int(conf["unit_size"] * conf["scale_mode"])
        elif type(conf["scale_mode"]) == int and conf["scale_mode"] > 0 and conf["scale_mode"]<=conf["num_clients"]:
            if cid < conf["scale_mode"]:
                unit_size = conf["unit_size"]
            else:
                unit_size = conf["unit_size"] // 2
        else:
            raise ValueError('scale mode not recognized{conf["scale_mode"]}')
    elif conf["ma_mode"] == "rm-cid":
        if type(conf["scale_mode"]) == float and conf["scale_mode"] <= 1.0:
            unit_size = int(conf["unit_size"] * conf["scale_mode"])
        elif type(conf["scale_mode"]) == int and conf["scale_mode"] >= 0 and conf["scale_mode"]<=conf["num_clients"]:
            if cid < conf["scale_mode"]:
                unit_size = conf["unit_size"]
            else:
                unit_size = conf["unit_size"] // 2
        elif type(conf["scale_mode"]) == int and conf["scale_mode"] < 0:
            unit_size = conf["unit_size"] + conf["scale_mode"]
        elif type(conf["scale_mode"]) == str:
            pattern1 = r'(\d+):(\d+)-(\d+):(\d+)'
            if conf["scale_mode"] == "basic":
                unit_size = conf["unit_size"] - 1
            elif conf["scale_mode"] == "long":
                if int(cid) in [0, 1, 2, 5, 6, 7]:
                    unit_size = conf["unit_size"] - 1
                else:
                    unit_size = conf["unit_size"] * 0.75
            elif re.match(pattern1, conf["scale_mode"]):
                match = re.match(pattern1, conf["scale_mode"])
                small_num = int(match.group(1))
                small_size = int(match.group(2))
                large_num = int(match.group(3))
                large_size = int(match.group(4))
                if int(cid) < small_num:
                    unit_size = small_size
                else:
                    unit_size = large_size
            else:
                raise ValueError('scale mode not recognized{conf["scale_mode"]}')
        else:
            raise ValueError('scale mode not recognized{conf["scale_mode"]}')
    elif conf["ma_mode"] == "no" or conf["ma_mode"]=="fjord":
        unit_size = conf["unit_size"]
    else:
        raise ValueError('model agnostic mode not recognized{conf["ma_mode"]}')
    return unit_size


def get_logger():
    logger = logging.getLogger("ma-fl-mia")
    logger.setLevel(logging.DEBUG)
    DEFAULT_FORMATTER = logging.Formatter(
    "%(levelname)s %(name)s %(asctime)s | %(filename)s:%(lineno)d | %(message)s"
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(DEFAULT_FORMATTER)
    logger.addHandler(console_handler)
    return logger

DEFAULT_LOGGER = get_logger()

def log(*args, **kwargs):
    DEFAULT_LOGGER.log(*args,**kwargs)

def create_nested_dict(flat_dict):
    nested_dict = {}

    for key, value in flat_dict.items():
        keys = key.split('_')
        current_dict = nested_dict

        for i, k in enumerate(keys[:-1]):
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]

        current_dict[keys[-1]] = value

    return nested_dict

def indices_of_largest_to_smallest(numbers):
    # Enumerate the numbers along with their indices
    enumerated_numbers = list(enumerate(numbers))
    
    # Sort the enumerated numbers in descending order of values
    sorted_indices = sorted(enumerated_numbers, key=lambda x: x[1], reverse=True)
    
    # Extract the indices from the sorted list
    indices = [index for index, _ in sorted_indices]
    
    return indices