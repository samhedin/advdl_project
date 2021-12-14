import torch
from torchvision import datasets
import torchvision.transforms as pth_transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

def plot_dataset(data, bins=255, filename=None, noise=None):
    d = []
    for batch,_ in data:
        if noise:
            batch = batch + torch.randn_like(batch) * noise
            d.extend([np.reshape(img.numpy(),(32*32)) * 1 for img in batch])
        else:
            d.extend([(np.reshape(img.numpy(),(32*32)) * 255).astype(np.uint8) for img in batch])
    d = np.array(d).reshape(-1)
    fig, ax = plt.subplots(figsize=(6, 4))
    color = "#0284C7" if noise is None else "#059669"
    plt.hist(d, bins=bins, color=color)
    plt.xlabel("Pixel value")

    scale_y = 1e6
    ticks_y = ticker.FuncFormatter(lambda x, pos: '{0:g}'.format(x / scale_y))
    ax.yaxis.set_major_formatter(ticks_y)
    plt.ylabel(r"Frequency ($\times 10^6)$")
    fig.savefig(filename, bbox_inches="tight")


def main():
    kwargs = {'num_workers':4, 'pin_memory':True, 'drop_last':True}
    noise = 0.3

    transforms = pth_transforms.Compose([pth_transforms.Grayscale(), pth_transforms.ToTensor()])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10("./data", train=True, 
        download=True, transform=transforms), batch_size=64, shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10("./data", train=False, 
                    transform=transforms), batch_size=64, shuffle=True, **kwargs)
    
    plot_dataset(train_loader, filename="images/CIFAR_10_noise01_dist.pdf", noise=0.1)


if __name__ == "__main__":
    main()
