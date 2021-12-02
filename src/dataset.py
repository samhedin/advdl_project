"""
Dataset loader
"""
from typing import Any
from scipy.ndimage import gaussian_filter
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as pth_transforms
import torchvision.datasets as datasets

# noise 0.5 taken from original paper.
def smooth(image, noise=0.1, proper_convolution=False):
    # Returns a tensor with the same size as input that is filled
    # with random numbers from a normal distribution with mean 0 and variance 1.
    if not proper_convolution:
        return image + torch.randn_like(image) * noise
    else:
        # https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html
        # or: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
        return gaussian_filter(image, sigma=1.2)

def build_dataset(config, noise=0.1, proper_convolution=False, smooth_output=False) -> Any:
    training_transform = pth_transforms.Compose([pth_transforms.ToTensor(), lambda img: smooth(img, noise, proper_convolution)])

    #the test data should sometimes be smooth, and sometimes regular.
    test_transform = pth_transforms.Compose([pth_transforms.ToTensor()]) if not smooth_output else training_transform

    if config.dataset == "MNIST":
        train_loader = DataLoader(
            datasets.MNIST(config.data_root, train=True, transform=training_transform, download=True),
            batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            datasets.MNIST(config.data_root, train=False, transform=test_transform, download=True),
            batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
    else:
        train_loader = DataLoader(
            datasets.CIFAR10(config.data_root, train=True, transform=training_transform, download=True),
            batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            datasets.CIFAR10(config.data_root, train=False, transform=test_transform, download=True),
            batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
    
    return train_loader, test_loader

# loads images produced by stage 1 to be used by stage 2
def stage2_image_loader(config):
    transform = pth_transforms.Compose([pth_transforms.ToTensor()])
    return DataLoader(datasets.ImageFolder("imgs/stage1_out", transform=transform), batch_size=config.batch_size, num_workers=4, pin_memory=True)
