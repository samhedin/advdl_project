"""
Callback for Weights & Biases
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only

from src.sampler import single_step_denoising
from src.utils import rescale_image, rescaling_inv


def get_wandb_logger(trainer: Trainer) -> WandbLogger:
    """Safely get Weights & Biases logger from trainer."""
    if isinstance(trainer.logger, WandbLogger):
        return trainer.logger

    if isinstance(trainer.logger, LoggerCollection):
        for logger in trainer.logger:
            if isinstance(logger, WandbLogger):
                return logger

    raise Exception(
        "You are using wandb related callback, but WandbLogger was not found for some reason..."
    )


class LogDebugSample(Callback):
    """Log the debug generated samples"""

    def __init__(self, n_logs: int = 4, denoising_mode: str = "single-step") -> None:
        super().__init__()
        self.n_logs = n_logs
        self.denoising_mode = denoising_mode
        self.ready = True

    def on_sanity_check_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.ready = False

    def on_sanity_check_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        """Start executing this callback only after all validation sanity checks end."""
        self.ready = True

    @rank_zero_only
    def on_validation_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        if not self.ready:
            return

        # With single-step denoising, we get both samples from the model and from the denoising step
        x_bar, x_tilde = None, None
        if self.denoising_mode == "single-step":
            x_b, x_t = single_step_denoising(
                pl_module,
                pl_module.hparams.sample_batch_size,
                noise=pl_module.hparams.noise,
            )
            x_bar = rescale_image(x_b).cpu().numpy()
            x_tilde = rescaling_inv(x_t).cpu().numpy()
