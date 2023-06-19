from tqdm import tqdm
import numpy as np
from src import models, attacks, utils
import os

from src.datasets import data_allocation
from src import datasets


def evaluate(conf, model, train_ds=None, val_ds=None, test_ds=None, verbose=1):
    """Attack on server"""
    if train_ds is None:
        train_ds, val_ds, test_ds = datasets.load_data(conf=conf)
        train_ds = datasets.preprocess_data(train_ds, conf)
        val_ds = datasets.preprocess_data(val_ds, conf)
        test_ds = datasets.preprocess_data(test_ds, conf)

    train_data = datasets.get_np_from_ds(train_ds)
    test_data = datasets.get_np_from_ds(test_ds)
    r = data_allocation.get_mia_datasets(
        train_data,
        test_data,
        conf["n_attacker_knowledge"],
        conf["n_attack_sample"],
        conf["seed"],
    )

    train_performance = models.evaluate(model, train_ds, conf, verbose=verbose)
    if val_ds is not None:
        val_performance = models.evaluate(model, val_ds, conf, verbose=verbose)
    else:
        val_performance = [None] * len(train_performance)
    test_performance = models.evaluate(model, test_ds, conf, verbose=verbose)
    mia_preds = attacks.attack(
        model,
        r["attacker_knowledge"],
        r["mia_data"],
        models.get_loss(),
        verbose=verbose,
    )
    results = {
        "test_acc": test_performance[1],
        "val_acc": val_performance[1],
        "train_acc": train_performance[1],
        "unit_size": conf["unit_size"],
        "alpha": conf["dirichlet_alpha"],
        "model_id": conf["model_id"],
        "params": models.count_params(model),
        "model_mode": conf["model_mode"],
        "scale_mode": conf["scale_mode"],
        "ma_mode": conf["ma_mode"],
    }
    for k, v in mia_preds.items():
        results[k] = attacks.calculate_advantage(r["mia_labels"], v)

    return results


def attack_on_clients(
    conf, X_split=None, Y_split=None, train_ds=None, val_ds=None, test_ds=None
):
    """Attack on client"""
    if train_ds is None or X_split is None:
        train_ds, val_ds, test_ds = datasets.load_data(conf=conf)
        if X_split is None:
            X_train, Y_train = datasets.get_np_from_ds(train_ds)
            conf["len_total_data"] = len(X_train)
            X_split, Y_split = data_allocation.split_data(
                X_train,
                Y_train,
                conf["num_clients"],
                split_mode=conf["split_mode"],
                mode="clients",
                seed=conf["seed"],
                dirichlet_alpha=conf["dirichlet_alpha"],
            )
        train_ds = datasets.preprocess_data(train_ds, conf)
        val_ds = datasets.preprocess_data(val_ds, conf)
        test_ds = datasets.preprocess_data(test_ds, conf)

    res = []
    for cid in tqdm(range(len(X_split))):
        model_root_path = conf["paths"]["models"]
        model_id = conf["model_id"]
        last_epoch = str(conf["rounds"])
        epoch_id = "saved_model_post_" + last_epoch
        model_path = os.path.join(
            model_root_path, model_id, "clients", str(cid), epoch_id
        )
        if not os.path.exists(model_path):
            if conf["active_fraction"]==1.0:
                print("Unexpected behavior: Client model missing")
        else:
            local_unit_size = utils.calculate_unit_size(cid, conf, len(X_split[cid]))
            conf["local_unit_size"] = local_unit_size
            model = models.init_model(
                unit_size=local_unit_size,
                conf=conf,
                model_path=model_path,
                keep_scaling=True,
                static_bn=True
            )
            train_c_ds = datasets.get_ds_from_np((X_split[cid], Y_split[cid]))
            train_c_ds = datasets.preprocess_data(train_c_ds, conf=conf)
            r = evaluate(conf, model, train_c_ds, val_ds, test_ds, verbose=0)
            r["cid"] = cid
            r["local_unit_size"] = local_unit_size
            res.append(r)

    all_train_acc = [a["train_acc"] for a in res]
    avgs = {
        "avg_train_acc": np.mean(all_train_acc),
        "std_train_acc": np.std(all_train_acc),
    }
    all_test_acc = [a["test_acc"] for a in res]
    avgs["avg_test_acc"] = np.mean(all_test_acc)
    avgs["std_test_acc"] = np.std(all_test_acc)
    adv_list = []
    for k in res[0].keys():
        if "adv" in k:
            adv_list.append(k)
    for adv in adv_list:
        all_adv = [a[adv] for a in res]
        avgs["avg_" + adv] = np.mean(all_adv)
        avgs["std_" + adv] = np.std(all_adv)

    return {"client_results": res, "average": avgs}


def evaluate_per_client(
    conf, model, X_split, Y_split, train_ds=None, val_ds=None, test_ds=None
):
    """Server model attack with client data"""

    if train_ds is None:
        train_ds, val_ds, test_ds = datasets.load_data(conf=conf)
    X_test, Y_test = datasets.get_np_from_ds(test_ds)

    test_ds = datasets.preprocess_data(test_ds, conf)

    r = data_allocation.get_mia_datasets_client_balanced(
        X_split,
        Y_split,
        X_test,
        Y_test,
        conf["n_attacker_knowledge"],
        conf["n_attack_sample"],
        conf["seed"],
    )
    results = []
    for X_client, Y_client in tqdm(zip(X_split, Y_split), total=len(X_split)):
        c_res = {}
        train_c_ds = datasets.get_ds_from_np((X_client, Y_client))
        train_c_ds = datasets.preprocess_data(train_c_ds, conf)

        train_performance = models.evaluate(model, train_c_ds, conf, verbose=0)
        c_res["train_acc"] = train_performance[1]

        len_data = min(len(X_client), len(X_test))
        c_res["data_size"] = len(X_client)
        X_ctest = np.concatenate((X_client[:len_data], X_test[:len_data]))
        Y_ctest = np.concatenate((Y_client[:len_data], Y_test[:len_data]))
        mia_true = [1.0] * len_data + [0.0] * len_data
        mia_true = np.array(mia_true)
        mia_preds = attacks.attack(
            model,
            r["attacker_knowledge"],
            (X_ctest, Y_ctest),
            models.get_loss(),
            verbose=0,
        )
        for k, v in mia_preds.items():
            c_res[k] = attacks.calculate_advantage(mia_true, v)
        results.append(c_res)
    return results
