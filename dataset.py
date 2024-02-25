import random

import numpy as np
import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


def pad_if_smaller(img, size, fill=0):
    min_size = min(img.size)
    if min_size < size:
        ow, oh = img.size
        padh = size - oh if oh < size else 0
        padw = size - ow if ow < size else 0
        img = F.pad(img, (0, 0, padw, padh), fill=fill)
    return img


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomResize:
    def __init__(self, min_size, max_size=None):
        self.min_size = min_size
        if max_size is None:
            max_size = min_size
        self.max_size = max_size

    def __call__(self, image, target):
        size = random.randint(self.min_size, self.max_size)
        image = F.resize(image, size, antialias=True)
        target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
        return image, target


class RandomHorizontalFlip:
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, image, target):
        if random.random() < self.flip_prob:
            image = F.hflip(image)
            target = F.hflip(target)
        return image, target


class RandomCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = pad_if_smaller(image, self.size)
        target = pad_if_smaller(target, self.size, fill=255)
        crop_params = T.RandomCrop.get_params(image, (self.size, self.size))
        image = F.crop(image, *crop_params)
        target = F.crop(target, *crop_params)
        return image, target


class CenterCrop:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, target):
        image = F.center_crop(image, self.size)
        target = F.center_crop(target, self.size)
        return image, target


class PILToTensor:
    def __call__(self, image, target):
        image = F.pil_to_tensor(image)
        target = torch.as_tensor(np.array(target), dtype=torch.int64)
        return image, target


class ToDtype:
    def __init__(self, dtype, scale=False):
        self.dtype = dtype
        self.scale = scale

    def __call__(self, image, target):
        if not self.scale:
            return image.to(dtype=self.dtype), target
        image = F.convert_image_dtype(image, self.dtype)
        return image, target


class Normalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target


def get_transform(is_train, is_eval):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    base_size = 520
    min_size = int(0.5*base_size)
    max_size = int(2.0*base_size)
    crop_size = 480
    assert not (is_train and is_eval), "is_train and is_eval cannot be True at the same time"

    if is_train:
        return Compose([
            RandomResize(min_size=min_size, max_size=max_size),
            RandomHorizontalFlip(),
            RandomCrop(crop_size),
            PILToTensor(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=mean, std=std),
        ])
    elif is_eval:
        return Compose([
            RandomResize(base_size),
            PILToTensor(),
            ToDtype(torch.float32, scale=True),
            Normalize(mean=mean, std=std),
        ])
    else:
        # for inference
        def preprocessing(img, target):
            img = F.pil_to_tensor(img)
            img = F.convert_image_dtype(img, torch.float32)
            img = F.normalize(img, mean=mean, std=std)

            size = F.get_dimensions(img)[1:]
            target = F.resize(target, size, interpolation=T.InterpolationMode.NEAREST)
            target = torch.as_tensor(np.array(target), dtype=torch.int64)
            return img, target

        return preprocessing
    

def get_dataset(is_train, is_eval):
    assert not (is_train and is_eval), \
        "is_train and is_eval cannot be True at the same time"
    from torchvision.datasets import VOCSegmentation
    if is_train:
        return VOCSegmentation(
            root="./data",
            year="2012",
            image_set="train",
            download=False,
            transforms=get_transform(is_train=True, is_eval=False)
        )
    elif is_eval:
        return VOCSegmentation(
            root="./data",
            year="2012",
            image_set="val",
            download=False,
            transforms=get_transform(is_train=False, is_eval=True)
        )
    else:
        return VOCSegmentation(
            root="./data",
            year="2012",
            image_set="val",
            download=False,
            transforms=get_transform(is_train=False, is_eval=False)
        )
    

if __name__ == "__main__":
    dataset = get_dataset(is_train=True, is_eval=False)
    image, target = dataset[0]
    print(image.size(), target.size())

    dataset = get_dataset(is_train=False, is_eval=True)
    image, target = dataset[0]
    print(image.size(), target.size())

    dataset = get_dataset(is_train=False, is_eval=False)
    image, target = dataset[0]
    print(image.size(), target.size())