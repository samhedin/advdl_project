from dataclasses import dataclass

import numpy as np

import torch
import src.pixelcnn as pixelcnn

torch.manual_seed(42)
import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils
from torchvision import utils as tfutils
from src.dataset import build_dataset, smooth, stage2_image_loader
from src.sampler import single_step_denoising


@dataclass
class Config:
    data_root: str = "./data"
    dataset: str = "CIFAR10"
    batch_size: int = 64


def showimg(image, filepath=None):
    print(image.shape)
    image = np.transpose(image, (1,2,0))
    plt.imshow(image)
    if filepath:
        plt.savefig(filepath)
    else:
        plt.show()


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

def save_sample_grid(cnn_helper):
    sample = cnn_helper.sample(sample_batch_size=2)

    sample = utils.rescaling_inv(sample)
    grid_img = tfutils.make_grid(sample)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig("imgs/sample.png")


def rescaling_inv(x):
    return .5 * x  + .5


def train_stage1():
    """
    Stage 1 Training: Learning the smooth distribution
    """
    args = utils.parser()
    train_loader_smooth, test_loader_smooth = build_dataset(config, noise=0.1, proper_convolution=False, smooth_output=True)
    model = pixelcnn.CNN_helper(args, train_loader_smooth, test_loader_smooth, pretrained=True)
    model.train()

    sample = model.sample(sample_batch_size=2)
    sample = utils.rescaling_inv(sample)
    grid_img = tfutils.make_grid(sample)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig("imgs/stage1_out.png")

def demonstrate_single_step_denoising(config, args):
    train_loader_smooth, test_loader_smooth = build_dataset(config, noise=0.1, proper_convolution=False, smooth_output=True)
    helper = pixelcnn.CNN_helper(args, train_loader_smooth, test_loader_smooth, pretrained=True)
    x, x_tilde = single_step_denoising(helper, 4)

    f = plt.figure()
    a = f.add_subplot(2, 1, 1)
    a.title.set_text("Before denoising")

    grid_img = tfutils.make_grid(x_tilde.cpu())
    plt.imshow(grid_img.permute(1, 2, 0))

    a = f.add_subplot(2, 1, 2)
    a.title.set_text("After single step denoising")
    grid_img = tfutils.make_grid(x.cpu())
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig("imgs/ssd.png")

if __name__ == "__main__":
    config = Config()
    args = utils.parser()
    demonstrate_single_step_denoising(config, args)
