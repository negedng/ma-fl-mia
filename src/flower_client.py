import flwr as fl
from flwr.common.logger import log
from logging import ERROR, INFO
import numpy as np
import os
from src import model_aggregation, models, utils
from src import datasets


class FlowerClient(fl.client.NumPyClient):
    """Client implementation using Flower federated learning framework"""

    def __init__(self, cid, conf):
        self.cid = cid
        self.conf = conf

    def init_model(self):
        self.calculate_unit_size()
        model = models.init_model(
            self.conf["local_unit_size"],
            conf=self.conf,
            model_path=None,
            static_bn=True,
        )
        self.model = model

    def calculate_unit_size(self):
        self.conf["local_unit_size"] = utils.calculate_unit_size(
            self.cid, self.conf, self.len_train_data
        )

    def load_data(self, X, Y, X_test, Y_test):
        self.train_len = len(X)
        self.test_len = len(X_test)
        self.train_data = (X, Y)
        self.test_data = (X_test, Y_test)
        self.len_train_data = len(X)

    def get_parameters(self, config):
        return models.get_weights(self.model)

    def set_parameters(self, weights, config):
        """set weights either as a simple update or model agnostic way"""
        if self.conf["ma_mode"] == "heterofl":
            cp_weights = model_aggregation.crop_weights(
                weights, models.get_weights(self.model)
            )
            models.set_weights(self.model, cp_weights)
        elif self.conf["ma_mode"] == "rm-cid":
            if "round_seed" in config.keys() and self.conf["permutate_cuts"]:
                rand = utils.get_random_permutation(
                    self.cid, self.conf["num_clients"], config["round_seed"]
                )
            else:
                rand = self.cid
            cp_weights = model_aggregation.crop_weights(
                weights, models.get_weights(self.model), conf=self.conf, rand=rand
            )
            models.set_weights(self.model, cp_weights)
        else:
            models.set_weights(self.model, weights)

    def fit(self, weights, config):
        """Flower fit passing updated weights, data size and additional params in a dict"""
        try:
            self.set_parameters(weights, config)

            train_ds = datasets.get_ds_from_np(self.train_data)
            if self.conf["aug"]:
                train_ds = datasets.aug_data(train_ds, conf=self.conf)
            train_ds = datasets.preprocess_data(train_ds, conf=self.conf, shuffle=True)

            if (
                self.conf["save_last_clients"] > 0
                and config["round"]
                > self.conf["rounds"] - self.conf["save_last_clients"]
            ):
                # save client models in last rounds
                save_path = os.path.join(
                    self.conf["paths"]["models"],
                    self.conf["model_id"],
                    "clients",
                    str(self.cid),
                    f'saved_model_pre_{str(config["round"])}',
                )
                models.save_model(self.model, save_path)

            history = models.fit(self.model, train_ds, self.conf)

            if np.isnan(
                history.history["loss"][-1]
            ):  # or np.isnan(history.history['val_loss'][-1]):
                raise ValueError("Warning, client has NaN loss")

            shared_metrics = {"client_id": self.cid, "loss": history.history["loss"]}

            client_weight = self.train_len if self.conf["weight_clients"] else 1

            if self.conf["hide_layers"] == "no":
                trained_weights = models.get_weights(self.model)
            elif self.conf["hide_layers"] == "yes":
                trained_weights = models.get_weights(self.model)
                total_rows = len(trained_weights)
                r = np.random.randint(total_rows)
                trained_weights[r] = weights[r]
                self.set_parameters(trained_weights, config)
            else:
                raise NotImplementedError(
                    f'unrecognized hide_layers in config:{self.conf["hide_layers"]}'
                )

            # Save model
            if (
                self.conf["save_last_clients"] > 0
                and config["round"]
                > self.conf["rounds"] - self.conf["save_last_clients"]
            ):
                # save client models in last rounds
                save_path = os.path.join(
                    self.conf["paths"]["models"],
                    self.conf["model_id"],
                    "clients",
                    str(self.cid),
                    f'saved_model_post_{str(config["round"])}',
                )
                models.save_model(self.model, save_path)

        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise RuntimeError("Client training terminated unexpectedly")

        return trained_weights, client_weight, shared_metrics

    def evaluate(self, weights, config):
        try:
            self.set_parameters(weights, config)
            test_ds = datasets.get_ds_from_np(self.test_data)
            test_ds = datasets.preprocess_data(test_ds, self.conf)

            # Local model eval
            loss, local_accuracy = models.evaluate(self.model, test_ds, verbose=0)
            # Global model eval
            test_ds = datasets.get_ds_from_np(self.test_data)
            test_ds = datasets.preprocess_data(test_ds, self.conf)
            g_model = models.init_model(
                self.conf["unit_size"], conf=self.conf, weights=weights
            )
            loss, accuracy = models.evaluate(g_model, test_ds, verbose=0)

            return (
                loss,
                self.test_len,
                {"local_accuracy": local_accuracy, "accuracy": accuracy},
            )
        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise RuntimeError("Client evaluate terminated unexpectedly")
