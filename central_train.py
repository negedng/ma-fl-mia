import os
import json
from datetime import datetime

from src.datasets import data_allocation
from src import datasets

from src import utils, models, attacks, metrics
from src.models import model_utils
from exp import setups




def train(conf, train_ds=None, val_ds=None, test_ds=None):
    conf["model_id"] = datetime.now().strftime("%Y%m%d-%H%M%S")
    print("Loading data")
    if train_ds is None:
        train_ds, val_ds, test_ds = datasets.load_data(conf=conf)
    print("loading model")
    model = model_utils.init_model(conf["unit_size"], conf=conf)
    models.print_summary(model)
    if conf["aug"]:
        train_ds = datasets.aug_data(train_ds, conf=conf)
    train_ds = datasets.preprocess_data(train_ds, conf=conf, shuffle=True)
    val_ds = datasets.preprocess_data(val_ds, conf=conf)
    print("start fit")
    history = models.fit(model, train_ds, conf=conf, verbose=1, validation_data=val_ds)

    os.makedirs(os.path.join(conf["paths"]["models"], conf["model_id"]), mode=0o777)
    
    with open(
        os.path.join(conf["paths"]["models"], conf["model_id"], "config.json"), "w"
    ) as f:
        json.dump(conf, f, indent=4)
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
    conf = utils.load_config()

    conf_changes = setups.get_experiment(args.exp, args.config)
    for k, v in conf_changes[0].items():
        conf[k] = v
    os.environ["CUDA_VISIBLE_DEVICES"] = conf["CUDA_VISIBLE_DEVICES"]
    conf["paths"]["models"] = conf["paths"]["models"].replace("federated","centralized")

    train_ds, val_ds, test_ds = datasets.load_data(conf=conf)

    for cc in conf_changes:
        for k, v in cc.items():
            conf[k] = v
        print(conf)   

        model, model_conf = train(conf, train_ds, val_ds, test_ds)
        models.save_model(model, os.path.join(model_conf["paths"]["models"], model_conf["model_id"], "saved_model/"))

        results = metrics.evaluate(model_conf, model)
        print(results)
        with open(
            os.path.join(
                model_conf["paths"]["models"], model_conf["model_id"], "tests.json"
            ),
            "w",
        ) as f:
            f.write(json.dumps(results))
