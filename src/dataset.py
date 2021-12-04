"""
Dataset loader
"""
from typing import Any
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as pth_transforms
import torchvision.datasets as datasets

def scale_img(img):
    """Scale the input to range [-1, 1]"""
    img = 2 * (img - img.min()) / (img.max() - img.min()) - 1
    return img

# noise 0.5 taken from original paper.
def smooth(image, noise=0.5):
    # Returns a tensor with the same size as input that is filled
    # with random numbers from a normal distribution with mean 0 and variance 1.
    return image + torch.randn_like(image) * noise

def smooth_and_rescale(img, noise):
    return scale_img(smooth(img, noise))

def build_dataset(config, noise=0.5, smoot_data=True, grayscale = False) -> Any:
    transforms = [pth_transforms.ToTensor()]
    if smoot_data:
        transforms.append(lambda img: smooth_and_rescale(img, noise))
    if grayscale:
        transforms.append(pth_transforms.Grayscale(num_output_channels=1))
    training_transform = pth_transforms.Compose(transforms)

    #the test data should sometimes be smooth, and sometimes regular.
    test_transform = pth_transforms.Compose([pth_transforms.ToTensor()]) if not smoot_data else training_transform

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
