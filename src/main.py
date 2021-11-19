from dataclasses import dataclass

import torch

from src.model import PixelCNN

torch.manual_seed(42)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import src.utils as utils

matplotlib.use("TkAgg")
from scipy.ndimage import gaussian_filter
from src.dataset import build_dataset


@dataclass
class Config:
    data_root: str = "./data"
    dataset: str = "CIFAR10"
    batch_size: int = 64


def showimg(image, filepath=None):
    image = np.array(image)
    print(image.shape)
    image = np.transpose(image, (1,2,0))
    plt.imshow(image)
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()


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


def pixelcnn_model():
    pixelcnnpp_pretrained = "pretrained/pixel-cnn-pp/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth"
    model = PixelCNN(nr_resnet=5, nr_filters=160)
    utils.load_part_of_model(model, pixelcnnpp_pretrained)
    return model

def show_image(train_loader):
    # show an image
    for images, labels in train_loader:
        img = images[0]
        showimg(img, filepath="imgs/before.png")

        after_img = smooth(img)
        showimg(after_img, filepath="imgs/after.png")

        after_img_conv = smooth(img, proper_convolution=True)
        showimg(after_img_conv, filepath="imgs/after_conv.png")
        break


if __name__ == "__main__":
    config = Config()
    train_loader, test_loader = build_dataset(config)
    model = pixelcnn_model()
