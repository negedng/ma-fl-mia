import tensorflow as tf
import numpy as np

from src.models.tensorflow_models.layer_utils import get_scaler, get_norm


def simple_CNN(unit_size, num_classes=10, input_shape=(32, 32, 3), static_bn=False):
    """Define the CNN model

    somewhere from Kaggle"""
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                unit_size // 2,
                (3, 3),
                padding="same",
                activation="relu",
                input_shape=input_shape,
            ),
            tf.keras.layers.BatchNormalization(trainable=not (static_bn)),
            tf.keras.layers.Conv2D(
                unit_size // 2, (3, 3), padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(trainable=not (static_bn)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(
                unit_size, (3, 3), padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(trainable=not (static_bn)),
            tf.keras.layers.Conv2D(
                unit_size, (3, 3), padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(trainable=not (static_bn)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(
                unit_size * 2, (3, 3), padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(trainable=not (static_bn)),
            tf.keras.layers.Conv2D(
                unit_size * 2, (3, 3), padding="same", activation="relu"
            ),
            tf.keras.layers.BatchNormalization(trainable=not (static_bn)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(unit_size * 2, activation="relu"),
            tf.keras.layers.BatchNormalization(trainable=not (static_bn)),
            tf.keras.layers.Dense(num_classes),
        ]
    )
    return model
