"""
"""
from typing import Any, Optional

import torch
import torch.optim as optim
import torch.autograd as autograd

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

pl.seed_everything(42)

from src.pixelcnn.model import PixelCNN
from src.pixelcnn.utils import mix_logistic_loss, sample_from_discretized_mix_logistic


class SmoothPixelCNNModule(pl.LightningModule):
    def __init__(
        self,
        nr_resnet: Optional[int] = 5,
        nr_filters: Optional[int] = 160,
        nr_logistic_mix: Optional[int] = 10,
        input_channels: Optional[int] = 3,
        lr: Optional[float] = 0.0002,
        lr_decay: Optional[float] = 0.9995,
        pretrained_weights: Optional[str] = None,
        device: Optional[Any] = torch.device("cpu"),
        sample_batch_size: Optional[int] = 2,
        image_dim: Optional[Any] = (3, 32, 32),
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

        if self.hparams.pretrained_weights is not None:
            self._load_model_pretrained(self.model, self.hparams.pretrained_weights)

        self.criterion = mix_logistic_loss
        self.sample_op = sample_from_discretized_mix_logistic

    def _load_model_pretrained(self, model, pretrained_weights):
        ckpt = torch.load(pretrained_weights)
        model.load_state_dict(ckpt)

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch) -> Any:
        input = batch
        output = self.forward(input)
        loss = self.criterion(input, output)
        return loss, output

    def sample(self, sample_batch_size=None):
        C, H, W = self.hparams.image_dim
        if not sample_batch_size:
            sample_batch_size = self.hparams.sample_batch_size

        data = torch.zeros(sample_batch_size, C, H, W)
        data.to(self.device)

        for i in range(H):
            for j in range(W):
                with torch.no_grad():
                    data_v = autograd.Variable(data).to(self.device)
                    out = self.model(data_v, sample=True)
                    out_sample = self.sample_op(out, self.hparams.nr_logistic_mix)
                    data[:, :, i, j] = out_sample.data[:, :, i, j]
        return data

    def training_step(self, batch: Any, batch_idx: int) -> STEP_OUTPUT:
        loss, _ = self.step(batch)

        self.log("train/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def validation_step(self, batch: Any, batch_idx: int) -> Optional[STEP_OUTPUT]:
        loss, _ = self.step(batch)

        self.log("val/loss", loss, on_step=True, on_epoch=True)
        return {"loss": loss}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=1, gamma=self.hparams.lr_decay
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
