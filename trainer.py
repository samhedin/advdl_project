import argparse

import torch
import pytorch_lightning as pl

from src.modules import SmoothPixelCNNModule


def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_arguent('-d', '--dataset', type=str,
                        default='cifar', help='Can be either cifar|mnist')
    parser.add_argument('-p', '--print_every', type=int, default=50,
                        help='how many iterations between print statements')
    parser.add_argument('-t', '--save_interval', type=int, default=1,
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
    parser.add_argument('-v', '--smooth', type=bool, default=True,
                        help='Smooth input data?')

    # model
    parser.add_argument('-q', '--nr_resnet', type=int, default=5,
                        help='Number of residual blocks per stage of the model')
    parser.add_argument('-n', '--nr_filters', type=int, default=160,
                        help='Number of filters to use across the model. Higher = larger model.')
    parser.add_argument('-m', '--nr_logistic_mix', type=int, default=10,
                        help='Number of logistic components in the mixture. Higher = more flexible model')
    parser.add_argument('-l', '--lr', type=float,
                        default=0.0002, help='Base learning rate')
    parser.add_argument('-e', '--lr_decay', type=float, default=0.999995,
                        help='Learning rate decay, applied every step of the optimization')
    parser.add_argument('-b', '--batch_size', type=int, default=64,
                        help='Batch size during training per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=100, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    parser.add_argument('-c', '--cuda', type=int, default=1,
                            help='Use CUDA?')
    return parser.parse_args()


def main(args: argparse.Namespace):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model_cfg = {
        "nr_resnet": args.nr_resnet,
        "nr_filters": args.nr_filters,
        "nr_logistic_mix": args.nr_logistic_mix,
        "lr": args.lr,
        "lr_decay": args.lr_decay,
        "device": device,
        "sample_batch_size": 25
    }
    print("Model config", model_cfg)
    smooth_module = SmoothPixelCNNModule(**model_cfg)


if __name__ == "__main__":
    args = parser()
    main(args)
