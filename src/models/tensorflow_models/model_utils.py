from tqdm import tqdm
import tensorflow as tf
import numpy as np

from src.models.tensorflow_models import scaler


gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.list_logical_devices('GPU')
    # print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)



custom_objects = {"Scaler": scaler.Scaler}

def get_optimizer(conf={}):
    if "optimizer" not in conf.keys():
        conf["optimizer"] = "SGD"
    if "learning_rate" not in conf.keys():
        conf["learing_rate"] = 0.1
    if "weight_decay" not in conf.keys():
        conf["weight_decay"] = 5e-4
    if conf["optimizer"]=="Adam":
        return tf.keras.optimizers.Adam(learning_rate=conf["learning_rate"], clipnorm=conf["clipnorm"])
    if conf["optimizer"]=="SGD":
        return tf.keras.optimizers.SGD(learning_rate=conf["learning_rate"],
                                       decay=conf["weight_decay"], clipnorm=conf["clipnorm"])
    raise NotImplementedError(f'Optim not recognized: {conf["optimizer"]}')


def get_loss(conf={}):
    return tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True
    )


def load_model_weights(model, model_path):
    model.load_weights(model_path).expect_partial()


def prepare_model(model, conf):
    model.compile(
        optimizer=get_optimizer(conf),
        loss=get_loss(conf),
        metrics=["sparse_categorical_accuracy"],
    )



def evaluate(model, data, conf, verbose=0):
    r = model.evaluate(data, verbose=verbose)

    return r


def fit(model, data, conf, verbose=0, validation_data=None, round_config=None, early_stopping=False):
    if round_config is not None:
        model.optimizer.lr.assign(round_config["learning_rate"])
    history = model.fit(data, epochs=conf["epochs"], verbose=verbose, validation_data=validation_data)
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


def calculate_loss(y_pred, y_true, loss_function, reduction='auto'):
    if reduction=='none' or reduction=='mean':
        red_func = tf.keras.losses.Reduction.NONE
    elif reduction=='sum':
        red_func = tf.keras.losses.Reduction.SUM
    elif reduction=='auto':
        red_func = tf.keras.losses.Reduction.AUTO
    old_red = loss_function.reduction
    loss_function.reduction = red_func
    losses = loss_function(y_pred, y_true)
    loss_function.reduction = old_red
    if reduction=='mean':
        losses = np.mean(losses)
    return losses


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

def count_params(model):
    return model.count_params()