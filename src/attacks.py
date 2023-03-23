import numpy as np

from src import utils


def yeom_mi_attack(losses, avg_loss):
    """YEOM et all's membership inference attack using pred loss"""
    memberships = (losses < avg_loss).astype(int)
    return memberships


def get_af():
    return yeom_mi_attack


def attacker_observation(model, attacker_knowledge, loss_function):
    """Attacker tests model with it's train data knowledge"""
    x_train_attacker, y_train_attacker = attacker_knowledge
    y_pred = model.predict(x_train_attacker, verbose=0)
    loss_train_attacker = loss_function(y_train_attacker, y_pred)
    loss_train_attacker = loss_train_attacker.numpy()
    
    return loss_train_attacker


def attack(model, attacker_knowledge, mia_data, loss_function, attack_function, *args, **kwargs):
    """Attacker performs attack"""
    
    loss_train_attacker = attacker_observation(model, attacker_knowledge, loss_function)
    
    x_mia_data, y_mia_data = mia_data
    loss_mia = utils.predict_losses(model, x_mia_data, y_mia_data, loss_function, *args, **kwargs)
    
    attack_pred = attack_function(loss_mia, loss_train_attacker)
    
    return attack_pred


def calculate_advantage(y_true, y_pred):
    """Two times the advantage over 50% random guess to get a 0-100 percentage-like result"""
    return 2*(len(y_true[y_true==y_pred])-len(y_true)/2)/len(y_true)*100
