import numpy as np


def yeom_mi_attack(losses, avg_loss):
    """YEOM et all's membership inference attack using pred loss"""
    memberships = (losses < avg_loss).astype(int)
    return memberships
