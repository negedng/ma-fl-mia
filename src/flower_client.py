import flwr as fl
from flwr.common.dp import add_gaussian_noise, clip_by_l2

from src.utils import log
from logging import ERROR, INFO
import numpy as np
import os
from src import model_aggregation, models, utils
from src.models import model_utils
from src import datasets
import copy

class FlowerClient(fl.client.NumPyClient):
    """Client implementation using Flower federated learning framework"""

    def __init__(self, cid, conf):
        self.cid = cid
        self.conf = conf
        self.channel_idx_list = None

    def init_model(self):
        self.calculate_unit_size()
        model = model_utils.init_model(
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
        # print(self.conf["local_unit_size"])

    def load_data(self, X, Y, X_test, Y_test):
        self.train_len = len(X)
        self.test_len = len(X_test)
        self.train_data = (X, Y)
        self.test_data = (X_test, Y_test)
        self.len_train_data = len(X)

    def get_parameters(self, config):
        return models.get_weights(self.model)

    def calculate_parameters(self, weights, config):
        """set weights either as a simple update or model agnostic way"""
        
        if "channel_idx_list" in config.keys():
            print("IDs from config")
            cp_weights = model_aggregation.crop_channels(weights, config["channel_idx_list"])
        elif self.conf["ma_mode"] == "heterofl":
            cp_weights = model_aggregation.select_channels(
                weights, models.get_weights(self.model), conf=self.conf, rand=0
            )
        elif self.conf["ma_mode"] == "rm-cid":
            if "round_seed" in config.keys() and self.conf["permutate_cuts"]:
                rand = config["round_seed"]
            else:
                rand = self.cid
                raise IndexError("Warning, obsolete 'rand' selection")
            if self.conf["cut_type"]=="maxgrad" or self.conf["cut_type"]=="softmaxgrad":
                train_ds = datasets.get_ds_from_np(self.train_data)
                if self.conf["aug"]:
                    train_ds = datasets.aug_data(train_ds, conf=self.conf)
                train_ds = datasets.preprocess_data(train_ds, conf=self.conf, shuffle=True)
                g_model = model_utils.init_model(
                        self.conf["unit_size"], conf=self.conf, weights=weights, static_bn=True
                    )
                grads = models.get_gradients(g_model, train_ds, self.conf, config)
                w_from_shape = [l.shape for l in weights]
                w_to_shape = [l.shape for l in models.get_weights(self.model)]
                idx_ret = model_aggregation.cut_idx_new(w_from_shape, w_to_shape, conf=self.conf, grads=grads, rand=rand)
                #print(self.cid, idx_ret[0][0])
                cp_weights = model_aggregation.crop_channels(weights, idx_ret)
                self.channel_idx_list = idx_ret
            else:
                w_from_shape = [l.shape for l in weights]
                w_to_shape = [l.shape for l in models.get_weights(self.model)]
                idx_ret = model_aggregation.cut_idx_new(w_from_shape, w_to_shape, conf=self.conf, rand=rand)
                #print(rand, idx_ret[0][0])
                cp_weights = model_aggregation.crop_channels(weights, idx_ret)
                #cp_weights = model_aggregation.select_channels(
                #    weights, models.get_weights(self.model), conf=self.conf, rand=rand
                #)
                self.channel_idx_list = idx_ret
        else:
            cp_weights = weights
        return cp_weights

    def set_parameters(self, weights, config):
        """set weights either as a simple update or model agnostic way"""
        cp_weights = self.calculate_parameters(weights, config)
        models.set_weights(self.model, cp_weights)

    def fit(self, weights, config):
        """Flower fit passing updated weights, data size and additional params in a dict"""
        # return self.get_parameters(config), 1, {"client_id": self.cid, "loss":-1}
        try:
            self.set_parameters(weights, config)
            
            # !TODO debug mode
            #ws = self.get_parameters(config)
            #if self.cid == 1:
            #    for i in range(len(ws)):
            #        ws[i] = ws[i] * 2
            #return ws, 1, {"client_id": self.cid, "loss":-1}

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

            history = models.fit(self.model, train_ds, self.conf, round_config=config)

            if np.isnan(
                history.history["loss"][-1]
            ):  # or np.isnan(history.history['val_loss'][-1]):
                raise ValueError("Warning, client has NaN loss")

            shared_metrics = {"client_id": self.cid, "loss": history.history["loss"]}
            if self.channel_idx_list:
                shared_metrics["channel_idx_list"] = self.channel_idx_list

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
            
            test_ds = datasets.get_ds_from_np(self.test_data)
            test_ds = datasets.preprocess_data(test_ds, self.conf)
            # Local model eval
            self.set_parameters(weights, config)
            local_loss, local_accuracy = models.evaluate(self.model, test_ds, self.conf, verbose=0)
            #local_loss, local_accuracy = 0,0
            if config["calculate_global"]:
                # Global model eval
                test_ds = datasets.get_ds_from_np(self.test_data)
                test_ds = datasets.preprocess_data(test_ds, self.conf)
                g_model = model_utils.init_model(
                    self.conf["unit_size"], conf=self.conf, weights=weights, static_bn=True
                )
                loss, accuracy = models.evaluate(g_model, test_ds, self.conf, verbose=0)
                test_len = self.test_len
            else:
                loss = 0
                test_len = 0
                accuracy = 0
            return (
                loss,
                test_len,
                {"local_cid": self.cid,
                 "local_rate":self.conf['local_unit_size']/self.conf['unit_size'], 
                 "local_loss":local_loss, 
                 "local_accuracy": local_accuracy, 
                 "server_accuracy": accuracy,
                 "server_loss":loss},
            )
        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise RuntimeError("Client evaluate terminated unexpectedly")


class DPWrapperClient(fl.client.NumPyClient):
    """Wrapper for configuring a Flower client for DP.
    https://github.com/adap/flower/blob/v1.5.0/src/py/flwr/client/dpfedavg_numpy_client.py"""

    def __init__(self, client: fl.client.NumPyClient) -> None:
        super().__init__()
        self.client = client

    def get_properties(self, config):
        return self.client.get_properties(config)

    def get_parameters(self, config):
        return self.client.get_parameters(config)

    def evaluate(self, parameters, config):
        return self.client.evaluate(parameters, config)

    def fit(
        self, parameters, config
    ):
        """Train the provided parameters using the locally held dataset.

        This method first updates the local model using the original parameters
        provided. It then calculates the update by subtracting the original
        parameters from the updated model. The update is then clipped by an L2
        norm and Gaussian noise is added if specified by the configuration.

        The update is then applied to the original parameters to obtain the
        updated parameters which are returned along with the number of examples
        used and metrics computed during the fitting process.
        """
        original_params = self.client.calculate_parameters(parameters, config)
        original_params = copy.deepcopy(original_params)
        # Getting the updated model from the wrapped client
        updated_params, num_examples, metrics = self.client.fit(parameters, config)

        # Update = updated model - original model
        update = [np.subtract(x, y) for (x, y) in zip(updated_params, original_params)]

        if "dpfedavg_clip_norm" not in config:
            raise Exception("Clipping threshold not supplied by the server.")
        if not isinstance(config["dpfedavg_clip_norm"], float):
            raise Exception("Clipping threshold should be a floating point value.")

        # Clipping
        update, clipped = clip_by_l2(update, config["dpfedavg_clip_norm"])

        if "dpfedavg_noise_stddev" in config:
            if not isinstance(config["dpfedavg_noise_stddev"], float):
                raise Exception(
                    "Scale of noise to be added should be a floating point value."
                )
            # Noising
            update = add_gaussian_noise(update, config["dpfedavg_noise_stddev"])

        for i, _ in enumerate(original_params):
            updated_params[i] = original_params[i] + update[i]

        # Calculating value of norm indicator bit, required for adaptive clipping
        if "dpfedavg_adaptive_clip_enabled" in config:
            if not isinstance(config["dpfedavg_adaptive_clip_enabled"], bool):
                raise Exception(
                    "dpfedavg_adaptive_clip_enabled should be a boolean-valued flag."
                )
            metrics["dpfedavg_norm_bit"] = not clipped

        return updated_params, num_examples, metrics

