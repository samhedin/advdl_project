"""
Stage 1 and 2 training modules
"""
from typing import Any

import torch
import torch.optim as optim
import pytorch_lightning as pl
pl.seed_everything(42)

from src.pixelcnn import PixelCNN


def init_pixelcnn(params: Any):
    pass


class PixelCNNTrainingModule(pl.LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        self.save_hyperparameters()

        self.model = init_pixelcnn(self.hparams)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams["lr"])
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.hparams["lr_decay"])
        return {"optimizer": optimizer, "scheduler": scheduler}