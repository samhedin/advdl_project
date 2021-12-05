import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

pl.seed_everything(1234)

from src.dataset import build_dataset
from src.modules import SmoothPixelCNNModule


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--data_root", type=str, default="data", help="Location for the dataset"
    )
    parser.add_argument(
        "-o",
        "--save_dir",
        type=str,
        default="models",
        help="Location for parameter checkpoints and samples",
    )
    parser.add_argument(
        "-d", "--dataset", type=str, default="cifar", help="Can be either cifar|mnist"
    )
    parser.add_argument(
        "-p",
        "--print_every",
        type=int,
        default=50,
        help="how many iterations between print statements",
    )
    parser.add_argument(
        "-t",
        "--save_interval",
        type=int,
        default=1,
        help="Every how many epochs to write checkpoint/samples?",
    )
    parser.add_argument(
        "-r",
        "--load_params",
        type=str,
        default=None,
        help="Restore training from previous model checkpoint?",
    )
    parser.add_argument(
        "-v", "--smooth", type=bool, default=True, help="Smooth input data?"
    )

    # model
    parser.add_argument(
        "-q",
        "--nr_resnet",
        type=int,
        default=5,
        help="Number of residual blocks per stage of the model",
    )
    parser.add_argument(
        "-n",
        "--nr_filters",
        type=int,
        default=160,
        help="Number of filters to use across the model. Higher = larger model.",
    )
    parser.add_argument(
        "-m",
        "--nr_logistic_mix",
        type=int,
        default=10,
        help="Number of logistic components in the mixture. Higher = more flexible model",
    )
    parser.add_argument(
        "-l", "--lr", type=float, default=0.0002, help="Base learning rate"
    )
    parser.add_argument(
        "-e",
        "--lr_decay",
        type=float,
        default=0.999995,
        help="Learning rate decay, applied every step of the optimization",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=64,
        help="Batch size during training per GPU",
    )
    parser.add_argument(
        "-x",
        "--max_epochs",
        type=int,
        default=10,
        help="How many epochs to run in total?",
    )
    parser.add_argument(
        "-g", "--noise", type=float, default=0.3, help="Sigma (noise) to add to data"
    )
    parser.add_argument("--loss_type", type=str, default="continuous")
    return parser.parse_args()


def main(args: argparse.Namespace):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Building dataset...")
    dataset_cfg = {
        "data_root": args.data_root,
        "batch_size": args.batch_size,
    }
    print("Dataset config", dataset_cfg)
    train_loader, test_loader = build_dataset(**dataset_cfg)

    print("Building model...")
    model_cfg = {
        "nr_resnet": args.nr_resnet,
        "nr_filters": args.nr_filters,
        "nr_logistic_mix": args.nr_logistic_mix,
        "noise": args.noise,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "device": device,
        "sample_batch_size": 25,
        "loss_type": args.loss_type,
    }
    print("Model config", model_cfg)
    smooth_module = SmoothPixelCNNModule(**model_cfg)

    # Setup Lightning callbacks
    lr_callback = LearningRateMonitor(logging_interval="epoch")
    ckpt_callback = ModelCheckpoint(verbose=True, every_n_epochs=1)
    gpus = 1 if torch.cuda.is_available() else None

    trainer = pl.Trainer(
        gpus=gpus, max_epochs=args.max_epochs, callbacks=[ckpt_callback, lr_callback]
    )

    print("Start training")
    trainer.fit(smooth_module, train_loader, test_loader)


if __name__ == "__main__":
    args = parser()
    main(args)
