import tensorflow as tf
import numpy as np


class Scaler(tf.keras.layers.Layer):
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def call(self, inputs, training=None):
        if training:
            return inputs / self.rate
        else:
            return inputs

custom_objects = {
    "Scaler": Scaler
}

def diao_CNN(model_rate=1, num_classes=10, input_shape=(32,32,3), training_phase=False, use_scaler=False, norm_mode="bn", default_hidden=[64, 128, 256, 512]):
    """Model following the diao et al paper.
       Emmiting LN, GN and IN as it is not straightforward to cast to TF,
       and the paper shows superiority of the BN"""
    def get_configurable_layers(use_scaler, norm_mode):
        if norm_mode == "bn":
            norm = tf.keras.layers.BatchNormalization(momentum=0.0, trainable= not(training_phase))
        else:
            norm = tf.keras.layers.Lambda(lambda x: x)
        if use_scaler:
            scaler = Scaler(scaler_rate)
        else:
            scaler = tf.keras.layers.Lambda(lambda x: x)
        return norm, scaler

    hidden_sizes = [int(np.ceil(model_rate * x)) for x in default_hidden]
    scaler_rate = model_rate / 1

    norm, scaler = get_configurable_layers(use_scaler, norm_mode)

    layers = []
    layers.append(tf.keras.layers.Conv2D(hidden_sizes[0], 3, padding='same', input_shape=input_shape))
    layers.append(scaler)
    layers.append(norm)
    layers.append(tf.keras.layers.ReLU())
    layers.append(tf.keras.layers.MaxPool2D(2))
    for i in range(len(hidden_sizes) - 1):
        norm, scaler = get_configurable_layers(use_scaler, norm_mode)

        layers.append(tf.keras.layers.Conv2D(hidden_sizes[i + 1], 3, padding='same'))
        layers.append(scaler)
        layers.append(norm)
        layers.append(tf.keras.layers.ReLU())
        layers.append(tf.keras.layers.MaxPool2D(2))
    layers = layers[:-1]
    layers.extend([tf.keras.layers.GlobalAvgPool2D(),
                    tf.keras.layers.Flatten(),
                    tf.keras.layers.Dense(num_classes)])
    model = tf.keras.Sequential(layers)
    return model
    

def simple_CNN(unit_size, num_classes=10, input_shape=(32,32,3), training_phase=False):
    """Define the CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(unit_size//2, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(trainable= not(training_phase)),
        tf.keras.layers.Conv2D(unit_size//2, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(training_phase)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(unit_size, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(training_phase)),
        tf.keras.layers.Conv2D(unit_size, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(training_phase)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(unit_size*2, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(training_phase)),
        tf.keras.layers.Conv2D(unit_size*2, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(training_phase)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(unit_size*2, activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(training_phase)),
        tf.keras.layers.Dense(num_classes)
    ])
    return model

def get_model(unit_size, model_mode=None, conf={}, *args, **kwargs):
    if (model_mode is None) and ("model_mode" in conf.keys()):
        model_mode = conf['model_mode']
    if model_mode=="simple_CNN":
        return simple_CNN(unit_size, *args, **kwargs)
    elif model_mode=="diao_CNN":
        if "local_unit_size" not in conf.keys():
            local_unit_size = unit_size
            default_unit_size = unit_size
        else:
            local_unit_size = unit_size
            default_unit_size = conf['unit_size']
        default_hidden = [default_unit_size,
                          default_unit_size*2,
                          default_unit_size*4,
                          default_unit_size*8]
        model_rate = float(local_unit_size)/float(default_unit_size)
        return diao_CNN(model_rate, default_hidden=default_hidden, use_scaler=True, *args, **kwargs)
    raise ValueError(f'Unknown model type{mode}')
    

def get_optimizer(*args, **kwargs):
    return tf.keras.optimizers.Adam(*args, **kwargs)


def get_loss(*args, **kwargs):
    return tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, *args, **kwargs)


def calculate_unit_size(cid, conf, len_train_data):
    if conf['ma_mode']=='heterofl':
        if conf['scale_mode']=='standard':
            if len_train_data > 2500:
                unit_size = conf['unit_size']
            elif len_train_data > 1250:
                unit_size = conf['unit_size'] // 2
            elif len_train_data >750:
                unit_size = conf['unit_size'] // 4
            else:
                unit_size = conf['unit_size'] // 8
        elif conf['scale_mode']=='basic':
            if len_train_data > 2500:
                unit_size = conf['unit_size']
            else:
                unit_size = conf['unit_size'] // 2
        elif conf["scale_mode"]=="1250":
            if len_train_data > 1250:
                unit_size = conf['unit_size']
            else:
                unit_size = conf['unit_size'] // 2
        elif conf["scale_mode"]=='no':
            unit_size = conf['unit_size']
        else:
        	raise ValueError('scale mode not recognized{conf["scale_mode"]}')
    elif conf['ma_mode'] == 'rm-cid':
        if type(conf['scale_mode'])==float and conf['scale_mode']<=1.0:
            unit_size = int(conf['unit_size'] * conf['scale_mode'])
        elif conf['scale_mode']=='basic':
            unit_size = conf['unit_size'] - 1
        elif conf['scale_mode']=='long':
            if int(cid) in [0,1,2,5,6,7]:
                unit_size = conf['unit_size']-1
            else:
                unit_size = conf['unit_size']*0.75
        else:
            raise ValueError('scale mode not recognized{conf["scale_mode"]}')
    elif conf['ma_mode']=='no':
        unit_size = conf['unit_size']
    else:
        raise ValueError('model agnostic mode not recognized{conf["ma_mode"]}') 
    return unit_size

