import flwr as fl
from flwr.common.logger import log
from logging import ERROR, INFO
from logging import WARNING
from typing import Callable, Dict, List, Optional, Tuple, Union
from flwr.server.client_proxy import ClientProxy
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

from src import models, ma_utils    

class SaveAndLogStrategy(fl.server.strategy.FedAvg):
    """Adding saving and logging to the strategy pipeline"""

    def __init__(self, conf, *args, **kwargs):
        self.conf = conf
        self.client_leaving_history = []
        self.aggregated_parameters = None
        self.best_loss = np.inf
        super().__init__(*args, **kwargs)

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
        if self.conf['ma_mode'] == 'heterofl':
            parameters_aggregated = ndarrays_to_parameters(ma_utils.aggregate_hetero(weights_results))
        else:
            parameters_aggregated = ndarrays_to_parameters(aggregate(weights_results))

        # Aggregate custom metrics if aggregation fn was provided
        metrics_aggregated = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [(res.num_examples, res.metrics) for _, res in results]
            metrics_aggregated = self.fit_metrics_aggregation_fn(fit_metrics)
        elif server_round == 1:  # Only log this warning once
            log(WARNING, "No fit_metrics_aggregation_fn provided")
        self.aggregated_parameters  = parameters_aggregated # Why can't I access this at eval?
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
        if self.aggregated_parameters is not None and aggregated_result[0] < self.best_loss:
            self.best_loss = aggregated_result[0]
            save_path = os.path.join(self.conf['paths']['models'],
                                     self.conf['model_id'],
                                     'saved_model_best')
            log(INFO, "Saving model to %s", save_path)
            aggregated_weights = fl.common.parameters_to_ndarrays(
                self.aggregated_parameters)
            model = models.get_model(unit_size=self.conf['unit_size'], conf=self.conf)
            model.compile(optimizer=models.get_optimizer(),
                          loss=models.get_loss())
            model.set_weights(aggregated_weights)
            model.save(save_path)
        if rnd == self.conf['rounds']:
            # end of training calls
            save_path = os.path.join(self.conf['paths']['models'],
                                     self.conf['model_id'],
                                     'saved_model')
            log(INFO, "Saving model to %s", save_path)
            aggregated_weights = fl.common.parameters_to_ndarrays(
                self.aggregated_parameters)
            model = models.get_model(unit_size=self.conf['unit_size'], conf=self.conf)
            model.compile(optimizer=models.get_optimizer(),
                          loss=models.get_loss())
            model.set_weights(aggregated_weights)
            model.save(save_path)
        return aggregated_result

