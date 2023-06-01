import tensorflow as tf


class Scaler(tf.keras.layers.Layer):
    def __init__(self, rate, keep_scaling=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate = rate
        self.keep_scaling = keep_scaling

    def call(self, inputs, training=None):
        if training:
            return inputs / self.rate
        else:
            if self.keep_scaling:
                return inputs / self.rate
            return inputs
