import torchvision
from torch.utils.data import random_split
import torch
import numpy as np
from PIL import Image
import copy
import json


def load_data(dataset_mode="CIFAR10", val_split=True, conf={}):
    """Load datasets into dataset object"""
    if "dataset" in conf.keys():
        dataset_mode = conf["dataset"]

    if "val_split" in conf.keys():
        val_split = conf["val_split"]
    if "seed" not in conf.keys():
        conf["seed"] = None

    if dataset_mode == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(
            "./dump/dataset", train=True, download=True
        )
        testset = torchvision.datasets.CIFAR10(
            "./dump/dataset", train=False, download=True
        )

        if val_split:
            len_val = len(trainset) / 20
            len_train = len(trainset) - len_val
            trainset, valset = random_split(
                trainset,
                [len_train, len_val],
                torch.Generator().manual_seed(conf["seed"]),
            )
        else:
            valset = copy.deepcopy(testset)

        return trainset, valset, testset
    if dataset_mode == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(
            "./dump/dataset", train=True, download=True
        )
        testset = torchvision.datasets.CIFAR100(
            "./dump/dataset", train=False, download=True
        )

        if val_split:
            len_val = len(trainset) / 20
            len_train = len(trainset) - len_val
            trainset, valset = random_split(
                trainset,
                [len_train, len_val],
                torch.Generator().manual_seed(conf["seed"]),
            )
        else:
            valset = copy.deepcopy(testset)

        return trainset, valset, testset   
    if dataset_mode=="FEMNIST":
        with open("./dump/dataset/femnist/data/test/all_data_0_niid_2_keep_300_test_9.json") as f:
            testset = json.load(f)
        data = None
        with open("./dump/dataset/femnist/data/train/all_data_0_niid_2_keep_300_train_9.json") as f:
            data = json.load(f)
        if val_split:

            len_val = len(data["users"]) / 20
            len_train = len(data["users"]) - len_val
            idx_list = data["users"]
            np.random.default_rng(seed=conf["seed"]).shuffle(idx_list)
            idx_train = idx_list[:int(len_train)]
            idx_val = idx_list[int(len_train):]
            trainset = femnist_filter(data, idx_train)
            valset = femnist_filter(data, idx_val)
        else:
            trainset = data
            valset = copy.deepcopy(testset)
            
        return trainset, valset, testset
    raise NotImplementedError(dataset_mode)


def femnist_filter(data, idx_list):
    ret_data = {"user_data":{}}
    idx_in_order = np.where(np.isin(np.array(data["users"]),np.array(idx_list)))
    ret_data["users"] = list(np.array(data["users"])[idx_in_order])
    ret_data["num_samples"] = list(np.array(data["num_samples"])[idx_in_order])
    for idx in ret_data["users"]:
        print(idx)
        ret_data["user_data"][idx] = data["user_data"][idx]
    return ret_data

def preprocess_data(data, conf, shuffle=False, cache=False):
    """From torch.utils.data.Dataset to DataLoader"""
    add_transforms = []
    add_transforms.append(torchvision.transforms.ToTensor())
    if not conf["data_normalize"]:
        if conf["dataset"]=="CIFAR10" or conf["dataset"]=="CIFAR100":
            add_transforms.append(torchvision.transforms.Lambda(lambda x: x * 255))
    if conf["data_centralize"]:
        #!TODO check these numbers
        if conf["dataset"]=="CIFAR10":
            add_transforms.append(
                torchvision.transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                )
            )
        elif conf["dataset"]=="CIFAR100":
            raise NotImplementedError("Norm params unknown")
        elif conf["dataset"]=="FEMNIST":
            raise NotImplementedError("Norm params unknown")
        else:
            raise NotImplementedError("Dataset unknown", conf["dataset"])

    if data.transform is None:
        data.transform = torchvision.transforms.Compose([])
    old_transforms = data.transform.transforms
    new_transforms = old_transforms + add_transforms
    data.transform.transforms = new_transforms

    ds = torch.utils.data.DataLoader(
        data, batch_size=conf["batch_size"], shuffle=shuffle
    )
    return ds


import numpy as np
def get_np_from_femnist(femnist_dataset, return_writers=False):
    x = []
    y = []
    w = []
    for k, v in femnist_dataset["user_data"].items():
        x += v["x"]
        y += v["y"]
        w += [k]*len(v["y"])
    x = np.array(x)
    x = x.reshape(-1, 28, 28)
    y = np.array(y)
    w = np.array(w)
    if return_writers:
        return x,y,w
    return x,y

def get_np_from_dataset(dataset):
    return np.array(dataset.data), np.array(dataset.targets)


def get_np_from_dataloader(dataloader):
    image_batches = []
    label_batches = []
    print("Warning, you really shouldn't call this get_np_from_dataloader")
    for images, labels in dataloader:
        image_batches.append(images.detach().numpy())
        label_batches.append(labels.detach().numpy())
    numpy_labels = np.concatenate(label_batches, axis=0)
    numpy_images = np.concatenate(image_batches, axis=0)
    numpy_images = np.transpose(numpy_images, (0, 2, 3, 1))
    return numpy_images, numpy_labels


def get_np_from_ds(data):
    if isinstance(data, torch.utils.data.DataLoader):
        return get_np_from_dataloader(data)
    return get_np_from_dataset(data)


class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, data, transform=None, target_transform=None):
        self.data = data[0]
        self.targets = data[1]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        arr = self.data[idx]
        if arr.dtype == np.float32 or arr.dtype == np.float64:
            arr = (arr * 255).astype(np.uint8)
        image = Image.fromarray(arr)
        label = self.targets[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_ds_from_np(data):
    return CustomImageDataset(data, transform=None, target_transform=None)
