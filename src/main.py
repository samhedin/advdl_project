from dataclasses import dataclass

import torch
torch.manual_seed(42)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from scipy import signal
from scipy.ndimage import gaussian_filter

from src.dataset import build_dataset

@dataclass
class Config:
    data_root: str = './data'
    dataset: str = "MNIST"
    batch_size: int = 64


def showimg(image, filepath=None):
    image = np.array(image)
    image = np.reshape(image, (28,28))
    plt.imshow(image, cmap="Greys_r")
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()

# noise 0.5 taken from original paper.
def smooth(image, noise=0.5, proper_convolution=False):
    # Returns a tensor with the same size as input that is filled
    # with random numbers from a normal distribution with mean 0 and variance 1.
    if not proper_convolution:
        return image + torch.randn_like(image) * noise
    else:
        # https://scipy-lectures.org/intro/scipy/auto_examples/solutions/plot_image_blur.html
        # or: https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html
        return gaussian_filter(image, sigma=1.5)

def main():
    config = Config()
    train_loader, test_loader = build_dataset(config)

    # show an image
    for images, labels in train_loader:
        img = images[0]
        showimg(img, filepath="imgs/before.png")

        after_img = smooth(img)
        showimg(after_img, filepath="imgs/after.png")


        after_img_conv = smooth(img, proper_convolution=True)
        showimg(after_img_conv, filepath="imgs/after_conv.png")
        break


if __name__ == '__main__':
    main()
