import torchvision
import torch

def aug_data(data, conf):
    """Augment dataset or dataloader"""
    if isinstance(data, torch.utils.data.DataLoader):
        dataset = data.dataset
    else:
        dataset = data
    t = dataset.transform
    if t is None:
        t = torchvision.transforms.Compose([])
    
    orig_list = t.transforms
    add_aug = []

    if conf["aug_crop"] > 0:
        add_aug.append(torchvision.transforms.RandomCrop(32, padding=conf['aug_crop']))
    if conf["aug_horizontal_flip"]:
        add_aug.append(torchvision.transforms.RandomHorizontalFlip())
    
    new_list = add_aug + orig_list
    t.transforms = new_list
    dataset.transform = t
    return data