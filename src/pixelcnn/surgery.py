import pdb
import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.datasets as datasets
import torchvision.transforms as pth_transforms
import torchvision.transforms.functional as F
import torchvision.utils as pth_utils

their_rescaling = lambda x: (x - 0.5) * 2.0
our_rescaling = lambda x: 2 * (x - x.min()) / (x.max() - x.min()) - 1

rescaling_inv = lambda x: 0.5 * x + 0.5

def smooth(img):
    img = img + torch.rand_like(img) * 0.3
    return our_rescaling(img)

def show(imgs, filename=None):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fix, axs = plt.subplots(ncols=len(imgs), nrows=1, squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
    if filename:
        plt.savefig(filename, bbox_inches="tight")


def main():
    print("Setting up data loader")
    transforms = pth_transforms.Compose([pth_transforms.ToTensor(), smooth])
    train_loader = torch.utils.data.DataLoader(
        datasets.CIFAR10("./data", train=True, download=True, transform=transforms),
        batch_size=5,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    for (img, _) in train_loader:
        # img = img + torch.randn_like(img) * 0.3
        grid = pth_utils.make_grid(img, nrow=1, padding=1)
        print(grid.shape)
        show(grid, "./our_noise.png")
        break


if __name__ == "__main__":
    main()
