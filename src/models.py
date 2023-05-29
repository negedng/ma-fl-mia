import tensorflow as tf
import numpy as np

kaiming_normal = tf.keras.initializers.VarianceScaling(scale=2.0, mode='fan_out', distribution='untruncated_normal')


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
        norm = tf.keras.layers.BatchNormalization(momentum=0.0, trainable= not(static_bn), name=name)
    elif norm_mode == "ln":
        norm = tf.keras.layers.LayerNormalization(axis=[1, 2, 3], name=name)
    else:
        norm = tf.keras.layers.Lambda(lambda x: x, name=name)
    return norm
       
            
def diao_CNN(model_rate=1, num_classes=10, input_shape=(32,32,3), static_bn=False, use_scaler=True, keep_scaling=False, norm_mode="bn", default_hidden=[64, 128, 256, 512]):
    """Model following the diao et al paper.
       Emmiting LN, GN and IN as it is not straightforward to cast to TF,
       and the paper shows superiority of the BN"""

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


def alexnet(unit_size=64, num_classes=10, input_shape=(32,32,3), static_bn=False, use_scaler=True, keep_scaling=False, model_rate=1.0):
    '''AlexNet for CIFAR10. FC layers are removed. Paddings are adjusted.
    Without BN, the start learning rate should be 0.01
    (c) YANG, Wei 
    https://github.com/bearpaw/pytorch-classification/tree/master
    '''

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


def basic_block(x, planes, stride=1, downsample=None, name=None, use_scaler=False, scaler_rate=1.0, keep_scaling=False, norm_mode="bn", static_bn=False):
    identity = x

    out = x
    # out = conv3x3(x, planes, stride=stride, name=f'{name}.conv1')
    out = tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.conv1', padding="same")(out)
    out = get_scaler(use_scaler, scaler_rate, keep_scaling, name=f'{name}.scaler1')(out)
    out = get_norm(norm_mode, static_bn, name=f'{name}.bn1')(out)
    out = tf.keras.layers.ReLU(name=f'{name}.relu1')(out)

    # out = conv3x3(out, planes, name=f'{name}.conv2')
    out = tf.keras.layers.Conv2D(filters=planes, kernel_size=3, strides=1, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.conv2', padding="same")(out)
    out = get_scaler(use_scaler, scaler_rate, keep_scaling, name=f'{name}.scaler2')(out)
    out = get_norm("bn", False, name=f'{name}.bn2')(out)

    if downsample is not None:
        for layer in downsample:
            identity = layer(identity)

    out = tf.keras.layers.Add(name=f'{name}.add')([identity, out])
    out = tf.keras.layers.ReLU(name=f'{name}.relu2')(out)

    return out


def make_layer(x, planes, blocks, stride=1, name=None, use_scaler=False, scaler_rate=1.0, keep_scaling=False, norm_mode="bn", static_bn=False):
    downsample = None
    inplanes = x.shape[3]
    if stride != 1 or inplanes != planes:
        downsample = [
            tf.keras.layers.Conv2D(filters=planes, kernel_size=1, strides=stride, use_bias=False, kernel_initializer=kaiming_normal, name=f'{name}.0.downsample.0'),
            get_scaler(use_scaler, scaler_rate, keep_scaling, name=f'{name}.0.downsample.1'),
            get_norm(norm_mode, static_bn, name=f'{name}.0.downsample.2'),
        ]

    x = basic_block(x, planes, stride, downsample, name=f'{name}.0', use_scaler=use_scaler, scaler_rate=scaler_rate, keep_scaling=keep_scaling, norm_mode=norm_mode, static_bn=static_bn)
    for i in range(1, blocks):
        x = basic_block(x, planes, name=f'{name}.{i}', use_scaler=use_scaler, scaler_rate=scaler_rate, keep_scaling=keep_scaling, norm_mode=norm_mode, static_bn=static_bn)

    return x


def resnet(blocks_per_layer, unit_size=64, num_classes=1000, input_shape=(32,32,3), use_scaler=True, model_rate=1.0, keep_scaling=False, norm_mode="bn", static_bn=False):
    scaler_rate = model_rate

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    x = tf.keras.layers.Conv2D(filters=unit_size, kernel_size=7, strides=2, use_bias=False, kernel_initializer=kaiming_normal, padding='same', name='conv1')(x)
    x = get_scaler(use_scaler, scaler_rate, keep_scaling, name=f'scaler1')(x)
    x = get_norm(norm_mode, static_bn, name=f'bn1')(x)
    x = tf.keras.layers.ReLU(name='relu1')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2, padding='same', name='maxpool')(x)

    x = make_layer(x, unit_size, blocks_per_layer[0], name='layer1')
    x = make_layer(x, unit_size*2, blocks_per_layer[1], stride=2, name='layer2')
    x = make_layer(x, unit_size*4, blocks_per_layer[2], stride=2, name='layer3')
    x = make_layer(x, unit_size*8, blocks_per_layer[3], stride=2, name='layer4')

    x = tf.keras.layers.GlobalAveragePooling2D(name='avgpool')(x)
    initializer = tf.keras.initializers.RandomUniform(-1.0 / np.sqrt(512), 1.0 / np.sqrt(512))
    x = tf.keras.layers.Dense(units=num_classes, kernel_initializer=initializer, bias_initializer=initializer, name='fc')(x)
    outputs = x

    model = tf.keras.Model(inputs, outputs)
    return model


def resnet18(**kwargs):
    return resnet([2, 2, 2, 2], **kwargs)


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
    elif model_mode=="resnet18":
        model_rate = float(local_unit_size)/float(default_unit_size)
        return resnet18(unit_size=unit_size, model_rate=model_rate, *args, **kwargs)
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

