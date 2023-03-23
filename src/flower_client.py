import flwr as fl
from flwr.common.logger import log
from logging import ERROR, INFO
import numpy as np

from src import models

    

class FlowerClient(fl.client.NumPyClient):
    """Client implementation using Flower federated learning framework"""
    
    def __init__(self, cid, conf):
        self.cid = cid
        self.conf = conf
    
    def init_model(self):
        model = models.get_model(self.conf.unit_size)
        model.compile(optimizer=models.get_optimizer(learning_rate=self.conf.learning_rate),
                      loss=models.get_loss(),
                      metrics=['accuracy'])
        self.model = model
    
    def load_data(self, X, Y, X_test, Y_test):
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
  
    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, weights, config):
        """Flower fit passing updated weights, data size and additional params in a dict"""
        try:
            self.model.set_weights(weights)
            history = self.model.fit(
                self.X,
                self.Y,
                epochs=self.conf.epochs,
                batch_size=self.conf.batch_size,
                verbose=0
            )
            if np.isnan(history.history['loss'][-1]): # or np.isnan(history.history['val_loss'][-1]):
                raise ValueError("Warning, client has NaN loss")
            
            shared_metrics = {
                'client_id': self.cid,
                'loss': history.history['loss']
            }
        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise Error("Client training terminated unexpectedly")
        return self.model.get_weights(), len(self.X), shared_metrics

    def evaluate(self, weights, config):
        try:
            self.model.set_weights(weights)
            loss, _ = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
            return loss, len(self.X_test), {}
        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise Error("Client evaluate terminated unexpectedly")

