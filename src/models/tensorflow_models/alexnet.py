import tensorflow as tf
import numpy as np

from src.models.tensorflow_models.layer_utils import get_scaler, get_norm


def alexnet(unit_size=64, num_classes=10, input_shape=(32,32,3), static_bn=False, use_scaler=True, keep_scaling=False, model_rate=1.0):
    '''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
    Without BN, the start learning rate should be 0.01
    (c) YANG, Wei 
    https://github.com/bearpaw/pytorch-classification/tree/master
    '''

    scaler_rate = model_rate
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(unit_size, kernel_size=11, strides=4, padding='same', input_shape=input_shape, name="conv2d_1"),
        get_scaler(use_scaler, scaler_rate, keep_scaling, name="scaler_1"),
        tf.keras.layers.ReLU(name="re_lu_1"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', name="max_pooling2d_1"),
        tf.keras.layers.Conv2D(unit_size*3, kernel_size=5, padding='same', name="conv2d_2"),
        get_scaler(use_scaler, scaler_rate, keep_scaling, name="scaler_2"),
        tf.keras.layers.ReLU(name="re_lu_2"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', name="max_pooling2d_2"),
        tf.keras.layers.Conv2D(unit_size*6, kernel_size=3, padding='same', name="conv2d_3"),
        get_scaler(use_scaler, scaler_rate, keep_scaling, name="scaler_3"),
        tf.keras.layers.ReLU(name="re_lu_3"),
        tf.keras.layers.Conv2D(unit_size*4, kernel_size=3, padding='same', name="conv2d_4"),
        get_scaler(use_scaler, scaler_rate, keep_scaling, name="scaler_4"),
        tf.keras.layers.ReLU(name="re_lu_4"),
        tf.keras.layers.Conv2D(unit_size*4, kernel_size=3, padding='same', name="conv2d_5"),
        get_scaler(use_scaler, scaler_rate, keep_scaling, name="scaler_5"),
        tf.keras.layers.ReLU(name="re_lu_5"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', name="max_pooling2d_3"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(num_classes, name="output")
    ])
    return model

