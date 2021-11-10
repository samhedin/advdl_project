"""
Dataset loader
"""
from typing import Any

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as pth_transforms
import torchvision.datasets as datasets


def build_dataset(config) -> Any:
    transform = pth_transforms.Compose([pth_transforms.ToTensor()])

    if config.dataset == "MNIST":
        train_loader = DataLoader(
            datasets.MNIST(config.data_root, train=True, transform=transform, download=True),
            batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            datasets.MNIST(config.data_root, train=False, transform=transform, download=True),
            batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )

    if config.dataset == "CIFAR10":
        train_loader = DataLoader(
            datasets.CIFAR10(config.data_root, train=True, transform=transform, download=True),
            batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
        test_loader = DataLoader(
            datasets.CIFAR10(config.data_root, train=False, transform=transform, download=True),
            batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True
        )
    
    return train_loader, test_loader
