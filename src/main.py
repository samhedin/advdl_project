from dataclasses import dataclass

import torch
import src.pixelcnn_model as pixelcnn

torch.manual_seed(42)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import src.utils as utils
from torchvision import utils as tfutils
matplotlib.use("TkAgg")
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

def save_sample_grid():
    sample = cnn_help.sample(sample_batch_size=1)
    grid_img = tfutils.make_grid(sample)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig("imgs/sample.png")

if __name__ == "__main__":
    config = Config()
    train_loader, test_loader = build_dataset(config, noise=0.1, proper_convolution=False)
    args = utils.parser()

    cnn_help = pixelcnn.CNN_helper(args, train_loader, test_loader, pretrained=True)
    # cnn_help.train()
    save_sample_grid()
