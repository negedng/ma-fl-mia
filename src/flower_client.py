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
        if self.conf['ma_mode']=='heterofl':
            if self.conf['scale_mode']=='standard':
                if self.len_train_data > 2500:
                    unit_size = self.conf['unit_size']
                elif self.len_train_data > 1250:
                    unit_size = self.conf['unit_size'] // 2
                elif self.len_train_data >750:
                    unit_size = self.conf['unit_size'] // 4
                else:
                    unit_size = self.conf['unit_size'] // 8
            elif self.conf['scale_mode']=='basic':
                if self.len_train_data > 2500:
                    unit_size = self.conf['unit_size']
                else:
                    unit_size = self.conf['unit_size'] // 2
            elif self.conf["scale_mode"]=="1250":
                if self.len_train_data > 1250:
                    unit_size = self.conf['unit_size']
                else:
                    unit_size = self.conf['unit_size'] // 2
            elif self.conf["scale_mode"]=='no':
                unit_size = self.conf['unit_size']
            else:
            	raise ValueError('scale mode not recognized{self.conf["scale_mode"]}')
        elif self.conf['ma_mode'] == 'rm-cid':
            if type(self.conf['scale_mode'])==float and self.conf['scale_mode']<1.0:
                unit_size = self.conf['unit_size'] * self.conf['scale_mode'] 
            else:
                unit_size = self.conf['unit_size'] - 1
        else:
            unit_size = self.conf['unit_size'] 
        self.conf['local_unit_size'] = unit_size
    
    def load_data(self, X, Y, X_test, Y_test):
        self.X = X
        self.Y = Y
        self.X_test = X_test
        self.Y_test = Y_test
        self.len_train_data = len(X)
  
    def get_parameters(self, config):
        return self.model.get_weights()

    def set_parameters(self, weights):
        """set weights either as a simple update or model agnostic way"""
        if self.conf["ma_mode"] == "heterofl":
            cp_weights = ma_utils.crop_weights(weights, self.model.get_weights())
            self.model.set_weights(cp_weights)
        elif self.conf['ma_mode'] == 'rm-cid':
            cp_weights = ma_utils.crop_weights(weights, self.model.get_weights(), self.cid)
            self.model.set_weights(cp_weights)
        else:
            self.model.set_weights(weights)

    def fit(self, weights, config):
        """Flower fit passing updated weights, data size and additional params in a dict"""
        try:
            self.set_parameters(weights)
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
            self.set_parameters(weights)
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

