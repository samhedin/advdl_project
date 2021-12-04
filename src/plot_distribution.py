#!/usr/bin/env python3
from torch.utils.data import dataset
import cv2
from matplotlib import pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, utils as tfutils
from dataset import build_dataset, smooth, stage2_image_loader
from dataclasses import dataclass
import matplotlib
from skimage import color

@dataclass
class Config:
    data_root: str = "./data"
    dataset: str = "CIFAR10"
    batch_size: int = 64

matplotlib.use("TkAgg")

cifar10_regular, _ = build_dataset(Config(), smooth_output=False, grayscale=True)
cifar10_smooth, _ = build_dataset(Config(), smooth_output=True, grayscale=True)

def plot_dataset(data, bins=100, filename=None):
    d = []
    for batch,_ in data:
        d.extend([np.reshape(img.numpy(),(32*32)) for img in batch])
    d = np.array(d).reshape(-1)
    plt.hist(d, bins=bins, histtype="step")
    plt.savefig(filename)

# plot_dataset(cifar10_regular, filename="distribution_regular.png")
plot_dataset(cifar10_smooth, filename="distribution_smooth.png")
