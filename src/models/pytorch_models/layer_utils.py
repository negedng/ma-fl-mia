import torch.nn as nn
import torch

from src.models.pytorch_models.scaler import Scaler


def get_scaler(use_scaler, scaler_rate, keep_scaling):
    """Get scaler from config params"""
    if use_scaler:
        scaler = Scaler(scaler_rate, keep_scaling)
    else:
        scaler = nn.Identity()
    return scaler


def get_norm(unit_size, norm_mode, static_bn):
    """Get configurable norm mode"""
    if norm_mode == "bn":
        # From resnet:
        # norm = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=not(static_bn), name=name)
        # From diao:
        norm = nn.BatchNorm2d(
            unit_size, momentum=None, track_running_stats=not (static_bn)
        )
    elif norm_mode == "ln":
        norm = nn.GroupNorm(1, unit_size)
    elif norm_mode == "in":
        norm = nn.GroupNorm(unit_size, unit_size)
    elif norm_mode == "gn":
        norm = nn.GroupNorm(4, unit_size)
    else:
        norm = nn.Identity()
    return norm
