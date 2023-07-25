import flwr as fl
from flwr.common.logger import log
from logging import ERROR, INFO
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
from flwr.server.client_manager import ClientManager

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.strategy.aggregate import aggregate
import os
import numpy as np

from src import model_aggregation, models, utils
from src.models import model_utils

def fit_metrics_aggregation_fn(fit_metrics):
    losses = [a[1]["loss"] for a in fit_metrics]
    return {"avg_train_loss":np.mean(losses), "std_train_loss":np.std(losses)}

def evaluate_metrics_aggregation_fn(eval_metrics):
    eval_res = {
        "loss": sum([e[1]["loss"] for e in eval_metrics]),
        "accuracy": sum([e[1]["accuracy"] for e in eval_metrics]),
        "local_loss": np.mean([e[1]["local_loss"] for e in eval_metrics]),
        "local_accuracy": np.mean([e[1]["local_accuracy"] for e in eval_metrics]),
    }
    return eval_res

def get_example_model_shape(conf):
    model = model_utils.get_model_architecture(unit_size=conf["unit_size"], conf=conf, static_bn=True)
    weights = models.get_weights(model)
    shapes = [np.shape(l) for l in weights]
    return shapes

class SaveAndLogStrategy(fl.server.strategy.FedAvg):
    """Adding saving and logging to the strategy pipeline"""

    def __init__(self, conf, *args, **kwargs):
        self.conf = conf
        self.client_leaving_history = []
        self.aggregated_parameters = None
        self.best_loss = np.inf
        self.global_model_shapes = get_example_model_shape(conf)
        with open(
            os.path.join(
                self.conf["paths"]["models"], self.conf["model_id"], "log_history.csv"
            ),
            "w",
        ) as f:
            f.write("epoch,loss,accuracy,local_loss,local_accuracy\n")
        super().__init__(evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
                         fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
                         *args, **kwargs)

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate fit results using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Convert results
        weights_results = [
            (parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        for weights, n_samples in weights_results:
            l_shapes = []
            for layer in weights:
                l_shapes.append(np.shape(layer))
            #print(l_shapes)
        if self.conf["ma_mode"] == "heterofl":
            weights_aggregated = model_aggregation.aggregate_hetero(weights_results)
        elif self.conf["ma_mode"] == "rm-cid":
            cids = [fit_res.metrics["client_id"] for _, fit_res in results]
            # !TODO set same replacement here
            rands = utils.get_random_permutation_for_all(cids, server_round, len(cids), self.conf["permutate_cuts"])
            rands = [rands[cid] for cid in cids]
            
            weights_aggregated = model_aggregation.aggregate_rmcid(
                    weights_results,
                    rands,
                    self.global_model_shapes,
                    conf=self.conf,
                )
            
        else:
            weights_aggregated = aggregate(weights_results)

        parameters_aggregated = ndarrays_to_parameters(weights_aggregated)
        #import pdb
        #pdb.set_trace()
        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
            log(INFO, "aggregated fit results %s", str(metrics_aggregated))
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        self.aggregated_parameters = (
            parameters_aggregated  # Why can't I access this at eval?
        )
        return parameters_aggregated, metrics_aggregated

    def aggregate_evaluate(
        self,
        rnd,
        results,
        failures,
    ):
        aggregated_result = super().aggregate_evaluate(rnd, results, failures)
        log(
            INFO,
            "Aggregated results: %s",
            aggregated_result,
        )
        with open(
            os.path.join(
                self.conf["paths"]["models"], self.conf["model_id"], "log_history.csv"
            ),
            "a",
        ) as f:
            f.write(str(rnd))
            for k,v in aggregated_result[1].items():
                f.write(","+str(v))
            f.write("\n")
        if (
            self.aggregated_parameters is not None
            and aggregated_result[0] < self.best_loss
        ):
            self.best_loss = aggregated_result[0]
            save_path = os.path.join(
                self.conf["paths"]["models"], self.conf["model_id"], "saved_model_best"
            )
            log(INFO, "Saving model to %s", save_path)
            aggregated_weights = fl.common.parameters_to_ndarrays(
                self.aggregated_parameters
            )
            model = model_utils.init_model(
                self.conf["unit_size"], conf=self.conf, weights=aggregated_weights, static_bn=True
            )
            models.save_model(model, save_path)
        if rnd % 10 == 0:
            # save every 10th
            save_path = os.path.join(
                self.conf["paths"]["models"],
                self.conf["model_id"],
                f"saved_model_checkpoint",
            )
            log(INFO, "Saving model to %s", save_path)
            aggregated_weights = fl.common.parameters_to_ndarrays(
                self.aggregated_parameters
            )
            model = model_utils.init_model(
                self.conf["unit_size"], conf=self.conf, weights=aggregated_weights, static_bn=True
            )
            models.save_model(model, save_path)
        if rnd < self.conf["rounds"] and rnd >= (
            self.conf["rounds"] - self.conf["save_last_clients"]
        ):
            # for client analysis
            save_path = os.path.join(
                self.conf["paths"]["models"],
                self.conf["model_id"],
                f"saved_model_{str(rnd)}",
            )
            log(INFO, "Saving model to %s", save_path)
            aggregated_weights = fl.common.parameters_to_ndarrays(
                self.aggregated_parameters
            )
            model = model_utils.init_model(
                self.conf["unit_size"], conf=self.conf, weights=aggregated_weights, static_bn=True
            )
            models.save_model(model, save_path)
        if rnd == self.conf["rounds"]:
            # end of training calls
            save_path = os.path.join(
                self.conf["paths"]["models"], self.conf["model_id"], "saved_model"
            )
            log(INFO, "Saving model to %s", save_path)
            aggregated_weights = fl.common.parameters_to_ndarrays(
                self.aggregated_parameters
            )
            model = model_utils.init_model(
                self.conf["unit_size"], conf=self.conf, weights=aggregated_weights, static_bn=True
            )
            models.save_model(model, save_path)
        return aggregated_result

    def generate_client_config(self, round_seed: int, server_round:int) -> Dict:
        learning_rate = self.conf['learning_rate']
        if "scheduler" in self.conf.keys():
            last_update = str(max(int(x) for x in self.conf["scheduler"].keys() if int(x)<=server_round))
            learning_rate = learning_rate * self.conf["scheduler"][last_update]["learning_rate_reduction"]

        client_config = {'learning_rate': learning_rate,
                             'round_seed': round_seed,
                             'round': server_round}
        
        return client_config
       

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""

        # Sample clients
        sample_size, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )

        # Create custom configs
        n_clients = len(clients)
        self.last_n_clients = n_clients



        fit_configurations = []
        rands = utils.get_random_permutation_for_all([c.cid for c in clients],
                                                     server_round, 
                                                     self.last_n_clients, 
                                                     self.conf["permutate_cuts"])
        # print(rands)
        for client in clients:
            client_config = self.generate_client_config(round_seed=rands[client.cid], server_round=server_round)


            fit_configurations.append((client, FitIns(parameters, client_config)))

        if "scheduler" in self.conf.keys():
            if str(server_round) in self.conf["scheduler"].keys():
                log(INFO, f"Scheduler update, {client_config}")
                
        return fit_configurations

    def configure_evaluate(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        if self.fraction_evaluate == 0.0:
            return []
        config = {}
        evaluate_ins = EvaluateIns(parameters, config)

        # Sample clients
        sample_size, min_num_clients = self.num_evaluation_clients(
            client_manager.num_available()
        )
        clients = client_manager.sample(
            num_clients=sample_size, min_num_clients=min_num_clients
        )
        evaluate_configurations = []
        rands = utils.get_random_permutation_for_all([c.cid for c in clients],
                                                     server_round, 
                                                     self.last_n_clients, 
                                                     self.conf["permutate_cuts"])
        # print(rands)
        for idx, client in enumerate(clients):
            
            client_config = self.generate_client_config(round_seed=rands[client.cid], server_round=server_round)
            if idx==0:
                client_config["calculate_global"] = True
            else:
                client_config["calculate_global"] = False

            evaluate_configurations.append((client, FitIns(parameters, client_config)))

        return evaluate_configurations

    def num_fit_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Return sample size and required number of clients."""
        num_clients = int(num_available_clients * self.fraction_fit)
        return max(num_clients, self.min_fit_clients), self.min_available_clients
    
    def num_evaluation_clients(self, num_available_clients: int) -> Tuple[int, int]:
        """Use a fraction of available clients for evaluation."""
        num_clients = int(num_available_clients * self.fraction_evaluate)
        return max(num_clients, self.min_evaluate_clients), self.min_available_clients
    

    
