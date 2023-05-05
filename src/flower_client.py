import flwr as fl
from flwr.common.logger import log
from logging import ERROR, INFO
import numpy as np
import tensorflow as tf
import os

from src import models, ma_utils, data_preparation

class FlowerClient(fl.client.NumPyClient):
    """Client implementation using Flower federated learning framework"""
    
    def __init__(self, cid, conf):
        self.cid = cid
        self.conf = conf
    
    def init_model(self):
        self.calculate_unit_size()
        model = models.get_model(unit_size=self.conf['local_unit_size'], static_bn=True, conf=self.conf)
        model.compile(optimizer=models.get_optimizer(learning_rate=self.conf['learning_rate']),
                      loss=models.get_loss(),
                      metrics=['accuracy'])
        self.model = model
    
    def calculate_unit_size(self):
        self.conf['local_unit_size'] = models.calculate_unit_size(self.cid, self.conf, self.len_train_data)
    
    def load_data(self, X, Y, X_test, Y_test):
        self.train_len = len(X)
        self.test_len = len(X_test) 
        self.train_ds = tf.data.Dataset.from_tensor_slices((X,Y))
        self.test_ds = tf.data.Dataset.from_tensor_slices((X_test,Y_test))
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
            cp_weights = ma_utils.crop_weights(weights, self.model.get_weights(), conf=self.conf, rand=rand)
            self.model.set_weights(cp_weights)
        else:
            self.model.set_weights(weights)

    def fit(self, weights, config):
        """Flower fit passing updated weights, data size and additional params in a dict"""
        try:
            self.set_parameters(weights, config)
            
            train_ds = self.train_ds.map(lambda x,y: data_preparation.preprocess_ds(x,y,self.conf))
            train_ds = train_ds.shuffle(5000).batch(self.conf['batch_size']).prefetch(tf.data.AUTOTUNE)
            
            if self.conf['save_last_clients']>0 and config['round']>self.conf['rounds']-self.conf['save_last_clients']:
                # save client models in last rounds
                save_path = os.path.join(self.conf['paths']['models'],
                                         self.conf['model_id'], 
                                         "clients",
                                         str(self.cid),
                                         f'saved_model_pre_{str(config["round"])}')
                self.model.save(save_path)
                
            history = self.model.fit(
                train_ds,
                epochs=self.conf['epochs'],
                verbose=0
            )
            
            if self.conf['save_last_clients']>0 and config['round']>self.conf['rounds']-self.conf['save_last_clients']:
                # save client models in last rounds
                save_path = os.path.join(self.conf['paths']['models'],
                                         self.conf['model_id'], 
                                         "clients",
                                         str(self.cid),
                                         f'saved_model_post_{str(config["round"])}')
                self.model.save(save_path)
                
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
        w = self.train_len if self.conf['weight_clients'] else 1
        return self.model.get_weights(), w, shared_metrics

    def evaluate(self, weights, config):
        try:
            self.set_parameters(weights, config)
            test_ds = self.test_ds.map(lambda x,y: data_preparation.preprocess_ds(x,y,self.conf))
            test_ds = test_ds.batch(self.conf['batch_size']).prefetch(tf.data.AUTOTUNE)
            
            # Local model eval
            loss, local_accuracy = self.model.evaluate(test_ds, verbose=0)
            # Global model eval
            g_model = models.get_model(unit_size=self.conf['unit_size'], conf=self.conf)
            g_model.set_weights(weights)
            g_model.compile(optimizer=models.get_optimizer(learning_rate=self.conf['learning_rate']),
                      loss=models.get_loss(),
                      metrics=['accuracy'])                      
            loss, accuracy = g_model.evaluate(test_ds, verbose=0)
            
            return loss, self.test_len, {"local_accuracy": local_accuracy, "accuracy": accuracy}
        except Exception as e:
            log(
                ERROR,
                "Client error: %s",
                str(e),
            )
            raise Error("Client evaluate terminated unexpectedly")

