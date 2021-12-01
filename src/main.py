from dataclasses import dataclass

import torch
import src.pixelcnn_model as pixelcnn

torch.manual_seed(42)
import matplotlib.pyplot as plt
import numpy as np
import src.utils as utils
from torchvision import utils as tfutils
from src.dataset import build_dataset, smooth

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

    rescaling_inv = lambda x : .5 * x  + .5
    sample = rescaling_inv(sample)
    grid_img = tfutils.make_grid(sample)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig("imgs/sample.png")


# Compare how it is to generate images from pixelcnn
# trained on smooth vs trained on non-smooth data.
def compare_smooth_and_unsmooth():
    train_loader_smooth, test_loader_smooth = build_dataset(config, noise=0.1, proper_convolution=False, smooth_data=True)
    train_loader_regular, test_loader_regular = build_dataset(config, noise=0.1, proper_convolution=False, smooth_data=False)
    args = utils.parser()

    smooth_model = pixelcnn.CNN_helper(args, train_loader_smooth, test_loader_smooth)
    smooth_model.train()
    sample = smooth_model.sample(sample_batch_size=2)

    rescaling_inv = lambda x : .5 * x  + .5
    sample = rescaling_inv(sample)
    grid_img = tfutils.make_grid(sample)
    plt.imshow(grid_img.permute(1, 2, 0))

    plt.savefig("imgs/smooth.png")

    unsmooth_model = pixelcnn.CNN_helper(args, train_loader_regular, test_loader_regular)
    sample = unsmooth_model.sample(sample_batch_size=2)
    rescaling_inv = lambda x : .5 * x  + .5
    sample = rescaling_inv(sample)
    grid_img = tfutils.make_grid(sample)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig("imgs/unsmooth.png")

if __name__ == "__main__":
    config = Config()
    compare_smooth_and_unsmooth()
