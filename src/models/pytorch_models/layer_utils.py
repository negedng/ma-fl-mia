import torch.nn as nn
import torch

from src.models.pytorch_models.scaler import Scaler
from src.models.pytorch_models.od_layers import ODConv2d, ODLinear, ODBatchNorm2d, ODGroupNorm

def get_linear(ordered_dropout=False, p=None, *args, **kwargs):
    if ordered_dropout:
        return ODLinear(*args, **kwargs)
    return nn.Linear(*args, **kwargs)


def get_conv(ordered_dropout=False, p=None, *args, **kwargs):
    if ordered_dropout:
        return ODConv2d(*args, **kwargs)
    return nn.Conv2d(*args, **kwargs)


def get_scaler(use_scaler, scaler_rate, keep_scaling):
    """Get scaler from config params"""
    if use_scaler:
        scaler = Scaler(scaler_rate, keep_scaling)
    else:
        scaler = nn.Identity()
    return scaler


def get_norm(unit_size, norm_mode, static_bn, ordered_dropout=False, p=None):
    """Get configurable norm mode"""
    if norm_mode == "bn":
        # From resnet:
        # norm = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=not(static_bn), name=name)
        # From diao:
        if ordered_dropout:
            norm = ODBatchNorm2d(
                p=p,
                num_features=unit_size, momentum=None, track_running_stats=not (static_bn)
            )
        else:
            norm = nn.BatchNorm2d(
                unit_size, momentum=None, track_running_stats=not (static_bn)
            )
    elif norm_mode == "ln":
        if ordered_dropout:
            norm = ODGroupNorm(num_groups=1, num_channels=unit_size, p=p)
        else:
            norm = nn.GroupNorm(1, unit_size)
    elif norm_mode == "in":
        if ordered_dropout:
            norm = ODGroupNorm(num_groups=unit_size, num_channels=unit_size, p=p, reduce_groups=True)
        else:
            norm = nn.GroupNorm(unit_size, unit_size)
    elif norm_mode == "gn":
        if ordered_dropout:
            norm = ODGroupNorm(num_groups=4, num_channels=unit_size, p=p)
        else:
            norm = nn.GroupNorm(4, unit_size)
    else:
        norm = nn.Identity()
    return norm
