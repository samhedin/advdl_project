from dataclasses import dataclass

import argparse
import torch
import src.model as model

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

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='cifar', help='Can be either cifar|mnist')
    parser.add_argument('-p', '--print_every', type=int, default=50,
                        help='how many iterations between print statements')
    parser.add_argument('-t', '--save_interval', type=int, default=10,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=2, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    parser.add_argument('-c', '--cuda', type=bool, default=False,
                            help='Use CUDA?')
    return parser.parse_args()

if __name__ == "__main__":
    config = Config()
    train_loader, test_loader = build_dataset(config)
    args = parser()
    network = model.pixelcnn_model()
    model.train(network, args, train_loader, test_loader)
