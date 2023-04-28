import flwr as fl
from flwr.common.logger import log
from logging import ERROR, INFO
import numpy as np

from src import models, ma_utils

class FlowerClient(fl.client.NumPyClient):
    """Client implementation using Flower federated learning framework"""
    
    def __init__(self, cid, conf):
        self.cid = cid
        self.conf = conf
    
    def init_model(self):
        self.calculate_unit_size()
        model = models.get_model(unit_size=self.conf['local_unit_size'], training_phase=True, conf=self.conf)
        model.compile(optimizer=models.get_optimizer(learning_rate=self.conf['learning_rate']),
                      loss=models.get_loss(),
                      metrics=['accuracy'])
        self.model = model
    
    def calculate_unit_size(self):
        self.conf['local_unit_size'] = models.calculate_unit_size(self.cid, self.conf, self.len_train_data)
    
    def load_data(self, X, Y, X_test, Y_test):
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.len_train_data = len(X)
  
    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, weights, config):
        """set weights either as a simple update or model agnostic way"""
        if self.conf["ma_mode"] == "heterofl":
            cp_weights = ma_utils.crop_weights(weights, self.model.get_weights())
            self.model.set_weights(cp_weights)
        elif self.conf['ma_mode'] == 'rm-cid':
            if 'round_seed' in config.keys() and self.conf['permutate_cuts']:
                rand = ma_utils.get_random_permutation(self.cid, self.conf['num_clients'], config['round_seed'])
            else:
                rand = self.cid
            cp_weights = ma_utils.crop_weights(weights, self.model.get_weights(), rand)
            self.model.set_weights(cp_weights)
        else:
            self.model.set_weights(weights)

    def fit(self, weights, config):
        """Flower fit passing updated weights, data size and additional params in a dict"""
        try:
            self.set_parameters(weights, config)
            history = self.model.fit(
                self.X,
                self.Y,
                epochs=self.conf['epochs'],
                batch_size=self.conf['batch_size'],
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
        w = len(self.X) if self.conf['weight_clients'] else 1
        return self.model.get_weights(), w, shared_metrics

    def evaluate(self, weights, config):
        try:
            self.set_parameters(weights, config)
            loss, local_accuracy = self.model.evaluate(self.X_test, self.Y_test, verbose=0)
            g_model = models.get_model(unit_size=self.conf['unit_size'], conf=self.conf)
            g_model.set_weights(weights)
            g_model.compile(optimizer=models.get_optimizer(learning_rate=self.conf['learning_rate']),
                      loss=models.get_loss(),
                      metrics=['accuracy'])
            loss, accuracy = g_model.evaluate(self.X_test, self.Y_test, verbose=0)
            return loss, len(self.X_test), {"local_accuracy": local_accuracy, "accuracy": accuracy}
        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise Error("Client evaluate terminated unexpectedly")

