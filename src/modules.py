"""
"""
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT

from src.pixelcnn.model import PixelCNN, mix_logistic_loss, sample_from_discretized_mix_logistic_inverse_CDF
from src.pixelcnn.utils import (
    discretized_mix_logistic_loss,
    sample_from_discretized_mix_logistic,
)


class SmoothPixelCNNModule(pl.LightningModule):
    def __init__(
        self,
        nr_resnet: Optional[int] = 5,
        nr_filters: Optional[int] = 160,
        nr_logistic_mix: Optional[int] = 10,
        input_channels: Optional[int] = 3,
        noise: Optional[float] = None,
        lr: Optional[float] = 0.0002,
        lr_decay: Optional[float] = 0.9995,
        pretrained_weights: Optional[str] = None,
        device: Optional[Any] = torch.device("cpu"),
        sample_batch_size: Optional[int] = 2,
        image_dim: Optional[Any] = (3, 32, 32),
        loss_type: Optional[str] = "continuous",
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

        self.model = PixelCNN(
            nr_resnet=self.hparams.nr_resnet,
            nr_filters=self.hparams.nr_filters,
            nr_logistic_mix=self.hparams.nr_logistic_mix,
            input_channels=self.hparams.input_channels,
        )
        self.to(device)
        self.model.to(self.hparams.device)

        # Load pretrained weights if available
        if self.hparams.pretrained_weights is not None:
            self._load_model_pretrained(self.model, self.hparams.pretrained_weights)
            print(f"PixelCNN++ weights loaded from {self.hparams.pretrained_weights}")

        if loss_type == "continuous":
            self.criterion = mix_logistic_loss
        elif loss_type == "discretize":
            self.criterion = discretized_mix_logistic_loss
        else:
            raise ValueError("Unsupported loss type " + loss_type)

        self.sample_op = sample_from_discretized_mix_logistic
        # self.sample_op = sample_from_discretized_mix_logistic_inverse_CDF

    def _load_model_pretrained(self, model, pretrained_weights):
        ckpt = torch.load(pretrained_weights)
        model.load_state_dict(ckpt)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch) -> Any:
        input_tensor, _ = batch
        # Add noise to the input
        if self.hparams.noise:
            input_tensor = input_tensor + torch.rand_like(input_tensor) * self.hparams.noise
        output = self.forward(input_tensor)
        loss = self.criterion(input_tensor, output)
        return loss, output

    def sample(self, sample_batch_size=None, data=None):
        sample_model = self.model
        sample_model.eval()

        C, H, W = self.hparams.image_dim
        if not sample_batch_size:
            sample_batch_size = self.hparams.sample_batch_size

        if data is None:
            data = torch.zeros(sample_batch_size, C, H, W)
            data = torch.rand_like(data)
        data.to(self.device)

        with torch.no_grad():
            for i in range(H):
                for j in range(W):
                    data_v = autograd.Variable(data).to(self.device)
                    out = self.forward(data_v)
                    out_sample = self.sample_op(out, self.hparams.nr_logistic_mix)
                    # out_sample = self.sample_op(data_v, sample_model, self.hparams.nr_logistic_mix, clamp=False, bisection_iter=20)
                    data[:, :, i, j] = out_sample.data[:, :, i, j]
        return data

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, _ = self.step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, _ = self.step(batch)

        self.log("val/loss", loss, on_step=False, on_epoch=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.hparams.lr_decay
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
