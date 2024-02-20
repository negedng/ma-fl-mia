from collections import OrderedDict
from typing import List
import torch
import numpy as np
from tqdm import tqdm
import os
import copy
from src.models.pytorch_models.od_layers import sample

def get_cpu():
    return torch.device("cpu")

def get_device(conf):
    if len(conf["CUDA_VISIBLE_DEVICES"]) > 0:
        if "client_resources" in conf.keys() and conf["client_resources"] is not None:
            if "num_gpus" in conf["client_resources"].keys():
                if conf["client_resources"]["num_gpus"] > 0:
                    device = torch.device("cuda")
                    return device
    return get_cpu()


def get_weights(model):
    #return [layer.detach().cpu().numpy() for layer in model.parameters()]
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_weights(model, weights: List[np.ndarray]):
    params_dict = zip(model.state_dict().keys(), weights)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=False)


def get_loss(conf={}):
    return torch.nn.functional.cross_entropy
    #return torch.nn.CrossEntropyLoss(reduction="sum")


def get_optimizer(params, conf={}):
    if "optimizer" not in conf.keys():
        conf["optimizer"]="Adam"
    if "learning_rate" in conf.keys():
        lr = conf["learning_rate"]
    else:
        lr = 0.001
    if conf["optimizer"]=="Adam":
        return torch.optim.Adam(params, lr=lr)
    if conf["optimizer"]=="SGD":
        return torch.optim.SGD(params, lr=lr)
    raise NotImplementedError(f'Optim not recognized {conf["optimizer"]}')

def evaluate(model, data, conf, verbose=0):
    model.eval()
    loss_fn = get_loss(conf)
    correct, total, loss = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
            outputs = model(images)
            loss += loss_fn(outputs, labels, reduction="sum").item()
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(data.dataset)
    accuracy = correct / total
    return loss, accuracy


def save_model(model, model_path):
    try:
        os.makedirs(os.path.join(model_path))
    except FileExistsError:
        pass # Not nice...
    torch.save(model.state_dict(), os.path.join(model_path,"torchmodel.pt"))


def load_model_weights(model, model_path):
    model.load_state_dict(torch.load(os.path.join(model_path, "torchmodel.pt")))


def print_summary(model):
    print(model)
    print("Total number of parameters:", count_params(model))
    print("Trainable parameters:", count_params(model, only_trainable=True))


def predict(model, X, verbose=0):
    model.eval()
    model.to(get_cpu())
    with torch.no_grad():
        images = np_to_tensor(X)
        #images = batch.to(get_cpu())
        outputs = model(images)
        return outputs


def predict_ds(model, dataloader, conf, apply_softmax=True):
    total_outputs = []
    labels_all = None
    model.eval()     # Optional when not using Model Specific layer
    model.to(get_device(conf))
    for images, labels in dataloader:
        l = labels.detach().numpy()
        if labels_all is None:
            labels_all = l
        else:
            labels_all = np.concatenate((labels_all,l))
        images, labels = images.to(get_device(conf)), labels.to(get_device(conf))      
        outputs = model(images)
        if apply_softmax:
            outputs = torch.nn.functional.softmax(torch.Tensor(outputs), -1)
        outputs = outputs.to('cpu').detach().numpy()
        total_outputs.extend(outputs)
    return np.array(total_outputs), np.array(labels_all)


def np_to_tensor(images):
    return torch.from_numpy(np.transpose(images,(0,3,1,2)))


def predict_losses(model, X, Y, loss_function, verbose=0.5):
    model.eval()
    model.to(get_cpu())
    with torch.no_grad():
        if len(X)<1000:
            images, labels = X, Y
            images = np_to_tensor(images)
            labels = torch.from_numpy(labels)
            outputs = model(images)
            losses = loss_function(outputs, labels, reduction='none').detach().numpy()
            return np.array(losses)
        losses = []
        for i in range(0,len(X),1000):
            images, labels = X[i:i+1000], Y[i:i+1000]
            images = np_to_tensor(images)
            labels = torch.from_numpy(labels)
            outputs = model(images)
            loss = loss_function(outputs, labels, reduction='none').detach().numpy()
            losses.extend(loss)
        return np.array(losses)


def get_losses(model, dataloader, loss_function, conf):
    p, l = predict_ds(model, dataloader, apply_softmax=False, conf=conf)
    ls = loss_function(torch.from_numpy(p), torch.from_numpy(l), reduction='none').detach().numpy()
    return ls


def calculate_loss(y_true, y_pred, loss_function, reduction='none'):
    if type(y_pred) is np.ndarray:
        y_pred = torch.from_numpy(y_pred)
    if type(y_true) is np.ndarray:
        if np.isscalar(y_true[0]):
            y_true = torch.from_numpy(y_true)
            y_true = torch.eye(y_pred.shape[1])[y_true]
        else:
            y_true = torch.from_numpy(y_true)
    #if reduction =='none':
    return loss_function(y_pred, y_true, reduction=reduction).detach().numpy()


def prepare_model(model, conf):
    model.to(get_device(conf))


class History:
    def __init__(self):
        self.history = {"loss": [], "accuracy": []}


class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def fit(model, data, conf, verbose=0, validation_data=None, round_config=None, early_stopping=False, distill_target_model=None):
    model.train() # switch to training mode
    history = History()
    if round_config is not None:
        conf["learning_rate"] = round_config["learning_rate"]
    if distill_target_model is not None:
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    else:
        optimizer = get_optimizer(model.parameters(), conf)
    loss_fn = get_loss(conf)
    if conf["proximal_mu"]!=0:
        global_params = copy.deepcopy(model).parameters()
    if conf["ma_mode"]=="fjord" and conf["cut_type"]=="random_round":
        new_seed = np.random.randint(2**32-1)
        model.set_ordered_dropout_channels(new_seed)      
    if early_stopping:
        early_stopper = EarlyStopper()
    for epoch in range(conf["epochs"]):
        if conf["ma_mode"]=="fjord" and conf["cut_type"]=="random_epoch":
            new_seed = np.random.randint(2**32-1)
            model.set_ordered_dropout_channels(new_seed)   
        iterator = data
        if verbose>0.8:
            iterator = tqdm(data, total=int(np.ceil(len(data.dataset)/conf["batch_size"])))
        correct, total, epoch_loss = 0, 0, 0.0
        for images, labels in iterator:
            images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
            if conf["ma_mode"]=="fjord":
                p = sample(conf)
                if conf["cut_type"]=="random_batch":
                    new_seed = np.random.randint(2**32-1)
                    model.set_ordered_dropout_channels(new_seed)
                model.set_ordered_dropout_rate(p)
            optimizer.zero_grad()
            outputs = model(images)
            if distill_target_model is not None:
                # Distillation towards this model
                distill_target_model.train()
                target_out = distill_target_model(images)
                loss = torch.nn.KLDivLoss(reduction='batchmean')(torch.nn.functional.log_softmax(outputs, dim=1), torch.nn.functional.softmax(target_out, dim=1))
                
            elif conf["proximal_mu"]!=0:
                proximal_term = 0
                for local_weights, global_weights in zip(model.parameters(), global_params):
                    proximal_term += (local_weights - global_weights).norm(2)
                loss = loss_fn(outputs, labels) + (conf["proximal_mu"] / 2) * proximal_term
            else:
                loss = loss_fn(outputs, labels)
            loss.backward()
            if conf['clipnorm'] is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), conf['clipnorm'])
            optimizer.step()
            # Metrics
            epoch_loss += loss.item() * images.size(0)
            total += labels.size(0)
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            del loss, outputs
        epoch_loss /= len(data.dataset)
        epoch_acc = correct / total

        if validation_data is not None:
            model.eval() # validation
            with torch.no_grad():
                correct, total, val_loss = 0, 0, 0.0
                for images, labels in validation_data:
                    images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
                    outputs = model(images)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item() * images.size(0)
                    total += labels.size(0)
                    correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
                    del loss, outputs
                val_loss /= len(validation_data.dataset)
                val_acc = correct / total
            model.train() # switch to training mode
        if verbose > 0:
            v_string = ''
            if validation_data is not None:
                v_string = f" val_loss:{val_loss}, val_acc:{val_acc}"
            print(f"Epoch {epoch+1}: loss:{epoch_loss}, acc:{epoch_acc}"+v_string)
        history.history["loss"].append(epoch_loss)
        history.history["accuracy"].append(epoch_acc)
        if early_stopping:
            if early_stopper.early_stop(val_loss):
                return history
    return history


def count_params(model, only_trainable=False):
    if only_trainable:
        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        pytorch_total_params = sum(p.numel() for p in model.parameters())
    return pytorch_total_params


def get_gradients(model, data, conf, round_config=None):
    model.train() # switch to training mode
    if round_config is not None:
        conf["learning_rate"] = round_config["learning_rate"]
    optimizer = get_optimizer(model.parameters(), conf)
    optimizer.zero_grad()
    loss_fn = get_loss(conf)
    i, (images, labels) = next(enumerate(data))
    images, labels = images.to(get_device(conf)), labels.to(get_device(conf))
    outputs = model(images)
    loss = loss_fn(outputs, labels)
    loss.backward()
    grads = [f.grad.data.detach().cpu().numpy() for f in model.parameters()]
    p_grad = np.sum(np.absolute(grads[0]), axis=(1,2,3))
    #print(p_grad)
    #import pdb
    #pdb.set_trace()
    # grads = [np.absolute(l) for l in grads]
    # grads = [np.sum(l, axis=(2,3)) if len(l.shape)==4 else l for l in grads]
    return grads