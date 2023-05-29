import tensorflow as tf
import numpy as np

from src.models import scaler, diao_cnn, alexnet, simple_cnn, resnet


custom_objects = {
    "Scaler": scaler.Scaler
}

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
        return simple_cnn.simple_CNN(unit_size, *args, **kwargs)
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
        return diao_cnn.diao_CNN(model_rate, default_hidden=default_hidden, norm_mode=norm_mode, *args, **kwargs)
    elif model_mode=="alexnet":
        model_rate = float(local_unit_size)/float(default_unit_size)
        return alexnet.alexnet(unit_size, model_rate=model_rate, *args, **kwargs)
    elif model_mode=="resnet18":
        model_rate = float(local_unit_size)/float(default_unit_size)
        return resnet.resnet18(unit_size=unit_size, model_rate=model_rate, *args, **kwargs)
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

