import flwr as fl
from flwr.common.logger import log
from flwr.common import ndarrays_to_parameters
from logging import ERROR, INFO
import shutil
from datetime import datetime
import os
import numpy as np
import json
import copy

from src import utils
from src import WANDB_EXISTS

global conf
conf = utils.load_config()
os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]


from src.datasets import data_allocation
from src import datasets
from src.flower_client import FlowerClient
from src.flower_strategy import SaveAndLogStrategy
from src import models, attacks, metrics
from src.models import model_utils
from exp import setups

# DP wrappers
from flwr.client import dpfedavg_numpy_client
from flwr.server.strategy import dpfedavg_adaptive


global X_split
global Y_split
global X_val
global Y_val



def client_fn(cid: str) -> fl.client.Client:
    """Prepare flower client from ID (following flower documentation)"""
    client = FlowerClient(int(cid), conf)
    client.load_data(X_split[int(cid)], Y_split[int(cid)], X_val, Y_val)
    client.init_model()
    if conf["dp"]:
        client = dpfedavg_numpy_client.DPFedAvgNumPyClient(client)
    return client


def train(conf, train_ds=None):
    global X_split
    global Y_split
    conf["model_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    os.makedirs(os.path.join(conf["paths"]["models"], conf["model_id"]), mode=0o777)
    if conf["save_last_clients"] > 0:
        os.makedirs(os.path.join(conf["paths"]["models"], conf["model_id"], "clients"))
        for i in range(conf["num_clients"]):
            os.makedirs(
                os.path.join(
                    conf["paths"]["models"], conf["model_id"], "clients", str(i)
                )
            )
    shutil.copy(
        conf["paths"]["code"],
        os.path.join(conf["paths"]["models"], conf["model_id"], "train.py"),
    )
    with open(
        os.path.join(conf["paths"]["models"], conf["model_id"], "config.json"), "w"
    ) as f:
        json.dump(conf, f, indent=4)
    
    if WANDB_EXISTS:
        import wandb
        wandb.init(
            project = "ma-fl-mia",
            tags = ["federated", conf["exp_name"]],
            config=conf
        )


    if train_ds is None:
        train_ds, _, _ = datasets.load_data(conf=conf)

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

    initial_model = model_utils.init_model(
        unit_size=conf["unit_size"],
        static_bn=True,
        conf=conf,
        model_path=conf["continue_from"],
    )
    models.print_summary(initial_model)
    ws = models.get_weights(initial_model)
    initial_parameters = ndarrays_to_parameters(
            ws
        )

    # Create FedAvg strategy
    strategy = SaveAndLogStrategy(
        conf=conf,
        initial_parameters=initial_parameters,  # avoid smaller models as init
        fraction_fit=conf["active_fraction"],  # Sample 10% of available clients for training
        fraction_evaluate=0.000001,  # Sample 5% of available clients for evaluation
        min_fit_clients=1,  # Never sample less than 10 clients for training
        min_evaluate_clients=1,  # Never sample less than 5 clients for evaluation
        # Wait until at least 75 clients are available
        # min_available_clients=1,
    )

    if conf["dp"]:
        if conf["dp_clipnorm"] is None:
            # Expects min-max normalized input
            strategy = dpfedavg_adaptive.DPFedAvgAdaptive(
                strategy=strategy,
                # clip_norm=conf['dp_clipnorm'],
                num_sampled_clients=conf["num_clients"],
                noise_multiplier=conf["dp_noise"],
            )
        else:
            strategy = dpfedavg_adaptive.DPFedAvgFixed(
                strategy=strategy,
                clip_norm=conf["dp_clipnorm"],
                num_sampled_clients=conf["num_clients"],
                noise_multiplier=conf["dp_noise"],
            )

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=conf["num_clients"],
        config=fl.server.ServerConfig(num_rounds=conf["rounds"]),
        strategy=strategy,
        ray_init_args=conf["ray_init_args"],
        client_resources=conf["client_resources"],
    )
    model_path = os.path.join(conf["paths"]["models"], conf["model_id"], "saved_model")
    model = model_utils.init_model(
        unit_size=conf["unit_size"], conf=conf, model_path=model_path, static_bn=True
    )
    return model, conf


if __name__ == "__main__":
    import argparse

    # Instantiate the parser
    parser = argparse.ArgumentParser(
        description="Select experiment from exp.setup, for single run change config.json"
    )
    parser.add_argument(
        "--exp", type=str, help="See full list in exp/setup.py", default="default"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="JSON str config changes, enclosed in ' characters ",
        default="{}",
    )
    args = parser.parse_args()
    conf_changes = setups.get_experiment(args.exp, args.config)
    for k, v in conf_changes[0].items():
        conf[k] = v

    train_ds, val_ds, test_ds = datasets.load_data(conf=conf)
    X_val, Y_val = datasets.get_np_from_ds(val_ds)

    f_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    conf["exp_name"] = f_name

    with open(
        os.path.join(os.path.dirname(conf["paths"]["code"]), f"dump/{f_name}.json"), "w"
    ) as f:
        f.write("[\n")

    for cc in conf_changes:
        for k, v in cc.items():
            conf[k] = v
        print(conf)
        train_ds, val_ds, test_ds = datasets.load_data(conf=conf)
        model, model_conf = train(conf, train_ds)

        print("Training completed, model evaluation")
        # Evaluate
        train_ds = datasets.preprocess_data(train_ds, conf)
        val_ds = datasets.preprocess_data(val_ds, conf, cache=True)
        test_ds = datasets.preprocess_data(test_ds, conf, cache=True)

        results = metrics.evaluate(model_conf, model, train_ds, val_ds, test_ds)

        # Per client eval
        res = metrics.attack_on_clients(
            model_conf, X_split, Y_split, train_ds, val_ds, test_ds
        )
        with open(
            os.path.join(
                model_conf["paths"]["models"],
                model_conf["model_id"],
                "client_results.json",
            ),
            "w",
        ) as f:
            f.write(json.dumps(res))

        results["client_attacks"] = res["average"]

        print(results)
        with open(
            os.path.join(
                model_conf["paths"]["models"], model_conf["model_id"], "tests.json"
            ),
            "w",
        ) as f:
            f.write(json.dumps(results))

        with open(
            os.path.join(os.path.dirname(conf["paths"]["code"]), f"dump/{f_name}.json"),
            "a",
        ) as f:
            f.write("  " + json.dumps(results) + ",\n")

    # endfor

    with open(
        os.path.join(os.path.dirname(conf["paths"]["code"]), f"dump/{f_name}.json"), "a"
    ) as f:
        f.write("\n]")
