import tensorflow as tf
from src.models.tensorflow_models.scaler import Scaler


def get_scaler(use_scaler, scaler_rate, keep_scaling, name=None):
    """Get scaler from config params"""
    if use_scaler:
        scaler = Scaler(scaler_rate, keep_scaling, name=name)
    else:
        scaler = tf.keras.layers.Lambda(lambda x: x, name=name)
    return scaler


def get_norm(norm_mode, static_bn, name=None):
    """Get configurable norm mode"""
    if norm_mode == "bn":
        # From resnet:
        # norm = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5, trainable=not(static_bn), name=name)
        # From diao:
        norm = tf.keras.layers.BatchNormalization(
            momentum=0.0, trainable=not (static_bn), name=name
        )
    elif norm_mode == "ln":
        norm = tf.keras.layers.LayerNormalization(axis=[1, 2, 3], name=name)
    else:
        norm = tf.keras.layers.Lambda(lambda x: x, name=name)
    return norm
