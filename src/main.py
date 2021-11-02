from dataclasses import dataclass

import torch
torch.manual_seed(42)

from src.dataset import build_dataset

@dataclass
class Config:
    data_root: str = './data'
    dataset: str = "MNIST"
    batch_size: int = 64

def main():
    config = Config()
    train_loader, test_loader = build_dataset(config)

    for images, labels in train_loader:
        import pdb; pdb.set_trace()
        break


if __name__ == '__main__':
    main()
