import tensorflow as tf
import numpy as np


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

custom_objects = {
    "Scaler": Scaler
}

def diao_CNN(model_rate=1, num_classes=10, input_shape=(32,32,3), static_bn=False, use_scaler=True, keep_scaling=False, norm_mode="bn", default_hidden=[64, 128, 256, 512]):
    """Model following the diao et al paper.
       Emmiting LN, GN and IN as it is not straightforward to cast to TF,
       and the paper shows superiority of the BN"""
    def get_configurable_layers(use_scaler, norm_mode, static_bn, keep_scaling, name=None):
        if name is not None:
            norm_name = 'norm_'+name
            scaler_name = 'scaler_'+name
        else:
            norm_name = None
            scaler_name = None
            
        if norm_mode == "bn":
            norm = tf.keras.layers.BatchNormalization(momentum=0.0, trainable= not(static_bn), name=norm_name)
        elif norm_mode == "ln":
            norm = tf.keras.layers.LayerNormalization(axis=[1, 2, 3], name=norm_name)
        else:
            norm = tf.keras.layers.Lambda(lambda x: x, name=norm_name)
        if use_scaler:
            scaler = Scaler(scaler_rate, keep_scaling, name=scaler_name)
        else:
            scaler = tf.keras.layers.Lambda(lambda x: x, name=scaler_name)
        return norm, scaler

    hidden_sizes = [int(np.ceil(model_rate * x)) for x in default_hidden]
    scaler_rate = model_rate

    norm, scaler = get_configurable_layers(use_scaler, norm_mode, static_bn, keep_scaling, name="1")

    layers = []
    layers.append(tf.keras.layers.Conv2D(hidden_sizes[0], 3, padding='same', input_shape=input_shape, name="conv2d_1"))
    layers.append(scaler)
    layers.append(norm)
    layers.append(tf.keras.layers.ReLU(name="re_lu_1"))
    layers.append(tf.keras.layers.MaxPool2D(2, name="max_pooling2d_1"))
    for i in range(len(hidden_sizes) - 1):
        norm, scaler = get_configurable_layers(use_scaler, norm_mode, static_bn, keep_scaling, name=str(i+2))

        layers.append(tf.keras.layers.Conv2D(hidden_sizes[i + 1], 3, padding='same', name="conv2d_"+str(i+2)))
        layers.append(scaler)
        layers.append(norm)
        layers.append(tf.keras.layers.ReLU(name="re_lu_"+str(i+2)))
        layers.append(tf.keras.layers.MaxPool2D(2, name="max_pooling2d_"+str(i+2)))
    layers = layers[:-1]
    layers.extend([tf.keras.layers.GlobalAvgPool2D(name="global_average_pooling2d"),
                    tf.keras.layers.Flatten(name="flatten"),
                    tf.keras.layers.Dense(num_classes, name="output")])
    model = tf.keras.Sequential(layers)
    return model


def alexnet(unit_size=64, num_classes=10, input_shape=(32,32,3), static_bn=False, use_scaler=True, keep_scaling=False, model_rate=1.0):
    '''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
    Without BN, the start learning rate should be 0.01
    (c) YANG, Wei 
    https://github.com/bearpaw/pytorch-classification/tree/master
    '''
    def get_scaler(use_scaler, keep_scaling, name=None):
        if use_scaler:
            scaler = Scaler(scaler_rate, keep_scaling, name=name)
        else:
            scaler = tf.keras.layers.Lambda(lambda x: x, name=name)
        return scaler
    scaler_rate = model_rate
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(unit_size, kernel_size=11, strides=4, padding='same', input_shape=input_shape, name="conv2d_1"),
        get_scaler(use_scaler, keep_scaling, name="scaler_1"),
        tf.keras.layers.ReLU(name="re_lu_1"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', name="max_pooling2d_1"),
        tf.keras.layers.Conv2D(unit_size*3, kernel_size=5, padding='same', name="conv2d_2"),
        get_scaler(use_scaler, keep_scaling, name="scaler_2"),
        tf.keras.layers.ReLU(name="re_lu_2"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', name="max_pooling2d_2"),
        tf.keras.layers.Conv2D(unit_size*6, kernel_size=3, padding='same', name="conv2d_3"),
        get_scaler(use_scaler, keep_scaling, name="scaler_3"),
        tf.keras.layers.ReLU(name="re_lu_3"),
        tf.keras.layers.Conv2D(unit_size*4, kernel_size=3, padding='same', name="conv2d_4"),
        get_scaler(use_scaler, keep_scaling, name="scaler_4"),
        tf.keras.layers.ReLU(name="re_lu_4"),
        tf.keras.layers.Conv2D(unit_size*4, kernel_size=3, padding='same', name="conv2d_5"),
        get_scaler(use_scaler, keep_scaling, name="scaler_5"),
        tf.keras.layers.ReLU(name="re_lu_5"),
        tf.keras.layers.MaxPool2D(pool_size=2, strides=2, padding='valid', name="max_pooling2d_3"),
        tf.keras.layers.Flatten(name="flatten"),
        tf.keras.layers.Dense(num_classes, name="output")
    ])
    return model


def simple_CNN(unit_size, num_classes=10, input_shape=(32,32,3), static_bn=False):
    """Define the CNN model"""
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(unit_size//2, (3,3), padding='same', activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(trainable= not(static_bn)),
        tf.keras.layers.Conv2D(unit_size//2, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(static_bn)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(unit_size, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(static_bn)),
        tf.keras.layers.Conv2D(unit_size, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(static_bn)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Conv2D(unit_size*2, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(static_bn)),
        tf.keras.layers.Conv2D(unit_size*2, (3,3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(static_bn)),
        tf.keras.layers.MaxPooling2D(pool_size=(2,2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(unit_size*2, activation='relu'),
        tf.keras.layers.BatchNormalization(trainable= not(static_bn)),
        tf.keras.layers.Dense(num_classes)
    ])
    return model

def get_model(unit_size, model_mode=None, conf={}, *args, **kwargs):

    if (model_mode is None) and ("model_mode" in conf.keys()):
        model_mode = conf['model_mode']
        
    if "local_unit_size" not in conf.keys():
        local_unit_size = unit_size
        default_unit_size = unit_size
    else:
        local_unit_size = unit_size
        default_unit_size = conf['unit_size']
    if model_mode=="simple_CNN":
        return simple_CNN(unit_size, *args, **kwargs)
    elif model_mode=="diao_CNN":
        if "norm_mode" not in conf.keys():
            norm_mode = "bn"
        else:
            norm_mode = conf['norm_mode']

        default_hidden = [default_unit_size,
                          default_unit_size*2,
                          default_unit_size*4,
                          default_unit_size*8]
        model_rate = float(local_unit_size)/float(default_unit_size)
        return diao_CNN(model_rate, default_hidden=default_hidden, norm_mode=norm_mode, *args, **kwargs)
    elif model_mode=="alexnet":
        model_rate = float(local_unit_size)/float(default_unit_size)
        return alexnet(unit_size, model_rate=model_rate, *args, **kwargs)
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
        elif type(conf['scale_mode'])==int and conf['scale_mode']<0:
            unit_size = conf['unit_size'] + conf['scale_mode']
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

