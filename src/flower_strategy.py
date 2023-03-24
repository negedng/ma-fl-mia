import flwr as fl
from flwr.common.logger import log
from logging import ERROR, INFO
import os
import numpy as np


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
        rnd,
        results,
        failures,
    ):
        """Aggregate model weights using weighted average and store best checkpoint"""
        aggregated_parameters_tuple = super().aggregate_fit(rnd, results, failures)
        self.aggregated_parameters, _ = aggregated_parameters_tuple

        return aggregated_parameters_tuple

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
            log(INFO, "Saving model")
            aggregated_weights = fl.common.parameters_to_ndarrays(
                self.aggregated_parameters)
            model = models.get_model(self.conf['unit_size'])
            model.compile(optimizer=models.get_optimizer(),
                          loss=models.get_loss())
            model.set_weights(aggregated_weights)
            model.save(os.path.join(conf['paths']['models'],
                       conf['model_id'], 'saved_model'))
        if rnd == self.conf['rounds']:
            # end of training calls
            pass
        return aggregated_result

