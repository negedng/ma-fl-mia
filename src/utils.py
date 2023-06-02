import numpy as np
import os
import json


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
        else:
            raise ValueError('scale mode not recognized{conf["scale_mode"]}')
    elif conf["ma_mode"] == "rm-cid":
        if type(conf["scale_mode"]) == float and conf["scale_mode"] <= 1.0:
            unit_size = int(conf["unit_size"] * conf["scale_mode"])
        elif type(conf["scale_mode"]) == int and conf["scale_mode"] < 0:
            unit_size = conf["unit_size"] + conf["scale_mode"]
        elif conf["scale_mode"] == "basic":
            unit_size = conf["unit_size"] - 1
        elif conf["scale_mode"] == "long":
            if int(cid) in [0, 1, 2, 5, 6, 7]:
                unit_size = conf["unit_size"] - 1
            else:
                unit_size = conf["unit_size"] * 0.75
        else:
            raise ValueError('scale mode not recognized{conf["scale_mode"]}')
    elif conf["ma_mode"] == "no":
        unit_size = conf["unit_size"]
    else:
        raise ValueError('model agnostic mode not recognized{conf["ma_mode"]}')
    return unit_size
