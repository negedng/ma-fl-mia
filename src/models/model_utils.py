
from . import get_alexnet, get_diao_CNN, get_resnet18, get_simple_CNN
from . import load_model_weights, set_weights, prepare_model
from src import TF_MODELS

def get_model_architecture(unit_size, model_mode=None, conf={}, *args, **kwargs):
    if ("dataset" not in conf.keys()) or conf["dataset"]=="CIFAR10":
        num_classes = 10
        if TF_MODELS:
            input_shape = (32,32,3)
        else:
            input_shape = (3,32,32)
    elif conf["dataset"]=="CIFAR100":
        num_classes = 100
        if TF_MODELS:
            input_shape = (32,32,3)
        else:
            input_shape = (3,32,32)       
    kwargs["num_classes"] = num_classes
    kwargs["input_shape"] = input_shape
    if (model_mode is None) and ("model_mode" in conf.keys()):
        model_mode = conf["model_mode"]

    if "local_unit_size" not in conf.keys():
        local_unit_size = unit_size
        default_unit_size = unit_size
    else:
        local_unit_size = unit_size
        default_unit_size = conf["unit_size"]
    
    model_rate = float(local_unit_size) / float(default_unit_size)
    ordered_dropout = conf["ma_mode"]=="fjord"
    if model_mode == "simple_CNN":
        return get_simple_CNN(unit_size, *args, **kwargs)
    elif model_mode == "diao_CNN":
        if "norm_mode" not in conf.keys():
            norm_mode = "bn"
        else:
            norm_mode = conf["norm_mode"]

        default_hidden = [
            default_unit_size,
            default_unit_size * 2,
            default_unit_size * 4,
            default_unit_size * 8,
        ]
        return get_diao_CNN(
            model_rate=model_rate,
            default_hidden=default_hidden,
            norm_mode=norm_mode,
            ordered_dropout=ordered_dropout,
            cut_type = conf["cut_type"],
            *args,
            **kwargs,
        )
    elif model_mode == "alexnet":
        return get_alexnet(unit_size, model_rate=model_rate, *args, **kwargs)
    elif model_mode == "resnet18":
        default_hidden = [
            default_unit_size,
            default_unit_size * 2,
            default_unit_size * 4,
            default_unit_size * 8,
        ]
        return get_resnet18(
            default_hidden=default_hidden, model_rate=model_rate, *args, **kwargs
        )
    raise ValueError(f"Unknown model type{model_mode}")


def init_model(unit_size, conf, model_path=None, weights=None, *args, **kwargs):
    model = get_model_architecture(unit_size=unit_size, conf=conf, *args, **kwargs)
    if model_path is not None:
        load_model_weights(model, model_path)
    if weights is not None:
        set_weights(model, weights)
    prepare_model(model, conf)
    return model
