"""
Callback for Weights & Biases
"""
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LoggerCollection, WandbLogger
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities import rank_zero_only


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

    def __init__(self, datamodule: pl.LightningDataModule, n_logs: int = 4) -> None:
        super().__init__()
        self.n_logs = n_logs
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
        
        samples = pl_module.sample(sample_batch_size=5)
        