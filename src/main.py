from dataclasses import dataclass

import torch
torch.manual_seed(42)
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use("TkAgg")

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

def smooth(image):
    return image

def main():
    config = Config()
    train_loader, test_loader = build_dataset(config)

    # show an image
    for images, labels in train_loader:
        img = images[0]
        showimg(img, filepath="imgs/before.png")
        after_img = smooth(img)
        showimg(after_img, filepath="imgs/after.png")
        break


if __name__ == '__main__':
    main()
