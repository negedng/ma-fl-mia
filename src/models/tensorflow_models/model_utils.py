from tqdm import tqdm
import tensorflow as tf
import numpy as np

from src.models.tensorflow_models import scaler, diao_cnn, alexnet, simple_cnn, resnet

print(tf.config.list_physical_devices('GPU'))

custom_objects = {"Scaler": scaler.Scaler}


def get_model_architecture(unit_size, model_mode=None, conf={}, *args, **kwargs):
    if (model_mode is None) and ("model_mode" in conf.keys()):
        model_mode = conf["model_mode"]

    if "local_unit_size" not in conf.keys():
        local_unit_size = unit_size
        default_unit_size = unit_size
    else:
        local_unit_size = unit_size
        default_unit_size = conf["unit_size"]
    if model_mode == "simple_CNN":
        return simple_cnn.simple_CNN(unit_size, *args, **kwargs)
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
        model_rate = float(local_unit_size) / float(default_unit_size)
        return diao_cnn.diao_CNN(
            model_rate,
            default_hidden=default_hidden,
            norm_mode=norm_mode,
            *args,
            **kwargs,
        )
    elif model_mode == "alexnet":
        model_rate = float(local_unit_size) / float(default_unit_size)
        return alexnet.alexnet(unit_size, model_rate=model_rate, *args, **kwargs)
    elif model_mode == "resnet18":
        model_rate = float(local_unit_size) / float(default_unit_size)
        return resnet.resnet18(
            unit_size=unit_size, model_rate=model_rate, *args, **kwargs
        )
    raise ValueError(f"Unknown model type{model_mode}")


def get_optimizer(*args, **kwargs):
    return tf.keras.optimizers.Adam(*args, **kwargs)


def get_loss(*args, **kwargs):
    return tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, *args, **kwargs
    )


def init_model(unit_size, conf, model_path=None, weights=None, *args, **kwargs):
    model = get_model_architecture(
        unit_size=unit_size, conf=conf, *args, **kwargs
    )
    if model_path is not None:
        model.load_weights(model_path).expect_partial()
    if weights is not None:
        model.set_weights(weights)
    model.compile(
        optimizer=get_optimizer(learning_rate=conf["learning_rate"]),
        loss=get_loss(),
        metrics=["sparse_categorical_accuracy"],
    )
    return model


def evaluate(model, data, conf, verbose=0):
    r = model.evaluate(data, verbose=verbose)

    return r


def fit(model, data, conf, verbose=0):
    history = model.fit(data, epochs=conf["epochs"], verbose=verbose)
    return history


def predict_losses(model, X, Y, loss_function, verbose=0.5):
    """Predict on model but returns with each individual loss"""
    losses = []

    p_verbose = 1.0 if verbose > 0.75 else 0.0
    Y_pred = model.predict(X, verbose=p_verbose)

    iterator = zip(Y, Y_pred)
    if verbose > 0.1:
        iterator = tqdm(iterator, total=len(Y))

    for y, y_pred in iterator:
        l = loss_function(y, y_pred)
        losses.append(l.numpy())
    return np.array(losses)


def predict(model, X, verbose=0):
    return model.predict(X, verbose=verbose)


def get_weights(model):
    return model.get_weights()


def set_weights(model, weights):
    model.set_weights(weights)


def save_model(model, model_path):
    model.save_weights(model_path)


def print_summary(model):
    print(model.summary())
