import tensorflow as tf
import numpy as np

from src.models.layer_utils import get_scaler, get_norm


def diao_CNN(model_rate=1, num_classes=10, input_shape=(32,32,3), static_bn=False, use_scaler=True, keep_scaling=False, norm_mode="bn", default_hidden=[64, 128, 256, 512]):
    """Model following the diao et al paper.
       Emmiting LN, GN and IN as it is not straightforward to cast to TF,
       and the paper shows superiority of the BN
       
       https://github.com/diaoenmao/HeteroFL-Computation-and-Communication-Efficient-Federated-Learning-for-Heterogeneous-Clients/tree/master
       """

    hidden_sizes = [int(np.ceil(model_rate * x)) for x in default_hidden]
    scaler_rate = model_rate

    layers = []
    layers.append(tf.keras.layers.Conv2D(hidden_sizes[0], 3, padding='same', input_shape=input_shape, name="conv2d_1"))
    layers.append(get_scaler(use_scaler, scaler_rate, keep_scaling, name="scaler_1"))
    layers.append(get_norm(norm_mode, static_bn, name="norm_1"))
    layers.append(tf.keras.layers.ReLU(name="re_lu_1"))
    layers.append(tf.keras.layers.MaxPool2D(2, name="max_pooling2d_1"))
    for i in range(len(hidden_sizes) - 1):
        layers.append(tf.keras.layers.Conv2D(hidden_sizes[i + 1], 3, padding='same', name="conv2d_"+str(i+2)))
        layers.append(get_scaler(use_scaler, scaler_rate, keep_scaling, name="scaler_"+str(i+2)))
        layers.append(get_norm(norm_mode, static_bn, name="norm_"+str(i+2)))
        layers.append(tf.keras.layers.ReLU(name="re_lu_"+str(i+2)))
        layers.append(tf.keras.layers.MaxPool2D(2, name="max_pooling2d_"+str(i+2)))
    layers = layers[:-1]
    layers.extend([tf.keras.layers.GlobalAvgPool2D(name="global_average_pooling2d"),
                    tf.keras.layers.Flatten(name="flatten"),
                    tf.keras.layers.Dense(num_classes, name="output")])
    model = tf.keras.Sequential(layers)
    return model

