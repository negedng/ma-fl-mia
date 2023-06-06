from collections import OrderedDict
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import os

from src.models.pytorch_models.diao_cnn import DiaoCNN


def get_cpu():
    return torch.device("cpu")

def get_device(conf):
    if len(conf["CUDA_VISIBLE_DEVICES"]) > 0:
        if "client_resources" in conf.keys():
            if "num_gpus" in conf["client_resources"].keys():
                if conf["client_resources"]["num_gpus"] > 0:
                    device = torch.device("cuda")
                    return device
    return get_cpu


def get_weights(model):
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_loss(*args, **kwargs):
    return torch.nn.functional.cross_entropy
    return torch.nn.CrossEntropyLoss(*args, **kwargs)


def get_optimizer(params, *args, **kwargs):
    if "learning_rate" in kwargs:
        lr = kwargs["learning_rate"]
    else:
        lr = 0.001
    return torch.optim.Adam(params, lr=lr)


def evaluate(model, data, conf, verbose=0):
    loss_fn = get_loss()
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in data:
            images, labels = batch[0].to(get_device(conf)), data[1].to(get_device(conf))
            outputs = model(images)
            loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(data.dataset)
    accuracy = correct / total
    return loss, accuracy


def save_model(model, model_path):
    torch.save(model.state_dict(), os.path.join(model_path,"torchmodel"))


def print_summary(model):
    print(model)


def predict(model, X, conf, verbose=0):
    model.to(get_cpu)
    ret = []
    with torch.no_grad():
        for batch in X:
            images = batch.to(get_cpu)
            outputs = model(images)
            ret.append(outputs)
    return ret


def predict_losses(model, X, Y, loss_function, verbose=0.5):
    model.to(get_cpu)
    all_losses = []
    with torch.no_grad():
        for batch in zip(X, Y):
            images, labels = batch[0].to(get_cpu), batch[1].to(get_cpu)
            outputs = model(images)
            losses = loss_function(outputs, labels).item()
            all_losses.append(losses)
    return np.array(all_losses)


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
        raise NotImplementedError(f"Not implemented model mode {model_mode}")
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
        model = DiaoCNN(
            model_rate=model_rate,
            default_hidden=default_hidden,
            norm_mode=norm_mode,
            *args,
            **kwargs,
        )
        return model
    elif model_mode == "alexnet":
        model_rate = float(local_unit_size) / float(default_unit_size)
        raise NotImplementedError(f"Not implemented model mode {model_mode}")
    elif model_mode == "resnet18":
        model_rate = float(local_unit_size) / float(default_unit_size)
        raise NotImplementedError(f"Not implemented model mode {model_mode}")
    raise ValueError(f"Unknown model type{model_mode}")
    return model


def init_model(unit_size, conf, model_path=None, weights=None, *args, **kwargs):
    model = get_model_architecture(unit_size=unit_size, conf=conf, *args, **kwargs)
    if model_path is not None:
        model.load_state_dict(torch.load(model_path))
    if weights is not None:
        set_weights(model, weights)
    model.to(get_device(conf))
    return model


class History:
    def __init__(self):
        self.history = {"loss": [], "accuracy": []}


def fit(model, data, conf, verbose=0):
    history = History()
    optimizer = get_optimizer(model.parameters(), learning_rate=conf["learning_rate"])
    loss_fn = get_loss()
    for epoch in range(conf["epochs"]):
        iterator = data
        if verbose>0.8:
            iterator = tqdm(data, total=int(np.ceil(len(data.dataset)/conf["batch_size"])))
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in iterator:
            images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss.item()
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            del loss, outputs
        epoch_loss /= len(data.dataset)
        epoch_acc = correct / total
        if verbose > 0:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, accuracy {epoch_acc}")
        history.history["loss"].append(epoch_loss)
        history.history["accuracy"].append(epoch_acc)
    return history
