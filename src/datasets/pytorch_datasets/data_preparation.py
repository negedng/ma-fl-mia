import torchvision
from torch.utils.data import random_split
import torch
import numpy as np
from PIL import Image


def load_data(dataset_mode="cifar10", val_split=True, conf={}):
    """Load datasets into dataset object"""

    if "val_split" in conf.keys():
        val_split = conf["val_split"]
    if "seed" not in conf.keys():
        conf["seed"] = None

    if dataset_mode == "cifar10":
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
            valset = testset

        return trainset, valset, testset

    raise NotImplementedError(dataset_mode)


def preprocess_data(data, conf, shuffle=False, cache=False):
    """From torch.utils.data.Dataset to DataLoader"""
    add_transforms = []
    add_transforms.append(torchvision.transforms.ToTensor())
    if not conf["data_normalize"]:
        add_transforms.append(torchvision.transforms.Lambda(lambda x: x * 255))
    if conf["data_centralize"]:
        #!TODO check these numbers
        add_transforms.append(
            torchvision.transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
            )
        )

    if data.transform is None:
        data.transform = torchvision.transforms.Compose([])
    old_transforms = data.transform.transforms
    new_transforms = old_transforms + add_transforms
    data.transform.transforms = new_transforms

    ds = torch.utils.data.DataLoader(
        data, batch_size=conf["batch_size"], shuffle=shuffle
    )

    return ds


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
    return CustomImageDataset(data)
