"""
Stage 2 modelling
"""
import time
import os
from pathlib import Path
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
from torchvision import utils as tutils
from utils import * 
from model import * 
import matplotlib.pyplot as plt
from functools import partial

def parser():
    parser = argparse.ArgumentParser()
    # data I/O
    parser.add_argument('-i', '--data_dir', type=str,
                        default='data', help='Location for the dataset')
    parser.add_argument('-o', '--save_dir', type=str, default='models',
                        help='Location for parameter checkpoints and samples')
    parser.add_argument('-d', '--dataset', type=str,
                        default='cifar', help='Can be either cifar|mnist')
    parser.add_argument('-p', '--print_every', type=int, default=50,
                        help='how many iterations between print statements')
    parser.add_argument('-t', '--save_interval', type=int, default=10, # Original: 10
                        help='Every how many epochs to write checkpoint/samples?')
    parser.add_argument('-r', '--load_params', type=str, default=None,
                        help='Restore training from previous model checkpoint?')
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
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='Batch size during training per GPU')
    parser.add_argument('-x', '--max_epochs', type=int,
                        default=1000, help='How many epochs to run in total?')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed to use')
    parser.add_argument('--exp_name', type=str, help="Name of the experiment")
    parser.add_argument('--smooth', type=bool, default=False, help="Whether to train on smoothed data")
    parser.add_argument('--resume_from', type=int, default=None, help="Epoch to resume training from")

    args = parser.parse_args()
    return args

args = parser()
# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

if args.exp_name:
    model_dir = Path("models") / args.exp_name
    model_dir.mkdir(parents=True, exist_ok=True)

    img_dir = Path("images") / args.exp_name
    img_dir.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")

sample_batch_size = 64
obs = (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
noise = 0.3
clean_noise = 0.01

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}_noise-{}'.format(args.lr, args.nr_resnet, args.nr_filters, str(noise).replace(".", ""))

loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

print("Creating model...")
model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.to(device)

if args.load_params:
    params = torch.load(args.load_params)
    added = 0
    for name, param in params.items():
        name = name.replace("module.", "")
        if name in model.state_dict().keys():
            model.state_dict()[name].copy_(param)
            added += 1

    print('added %s of params:' % (added / float(len(model.state_dict().keys()))))
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

def build_dataloaders():
    kwargs = {'num_workers':4, 'pin_memory':True, 'drop_last':True}
    print("Creating image loaders...")
    # Recale the image to range [-1, 1]
    ds_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

    return train_loader, test_loader

def sample(sample_model, in_data):
    stage1_images = torch.split(in_data, 32, dim=0)
    out_data = []
    for batch in stage1_images:
        # Sampling from stage 2 using inverse CDF
        x = torch.concat([batch, batch], dim=2)
        x = x.to(device)
        
        for i in range(-x.shape[-1], 0, 1):
            for j in range(x.shape[-1]):
                samples = sample_from_discretized_mix_logistic_inverse_CDF(
                    x, model=sample_model, nr_mix=10, clamp=False, bisection_iter=20)
                x[:, :, i, j] = samples[:, :, i, j]

        # img = rescaling_inv(x)[:, :, -batch_out.shape[-1]:, :]
        out_img = x[:, :, -x.shape[-1]:, :]
        out_data.append(out_img)
    
    out_data = torch.concat(out_data)
    return out_data


def train(model, train_loader, test_loader):
    print('starting training')

    for epoch in range(args.max_epochs):
        if args.resume_from:
            epoch += args.resume_from

        torch.cuda.synchronize()
        train_loss = 0.
        time_ = time.time()
        model.train(True)
        for batch_idx, (x,_) in enumerate(train_loader):
            x = x.to(device)  # input.cuda(non_blocking=True)
            # if args.smooth:
            noisy_x = x + torch.randn_like(x) * noise
            clean_x = x + torch.randn_like(x) * clean_noise
            x = torch.cat([noisy_x, clean_x], dim=2)
            x = Variable(x).to(device)
            output = model(x)[:, :, x.size()[-1]:, :]  # Take one 1 image from the output
            loss = loss_op(clean_x, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx + 1) % args.print_every == 0: 
                deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
                print('Epoch: {} - loss : {:.4f}, time : {:.4f}'.format(epoch, (train_loss / deno), (time.time() - time_)))
                train_loss = 0.
                time_ = time.time()

        # decrease learning rate
        scheduler.step()
        
        torch.cuda.synchronize()
        sample_model = partial(model, sample=True)
        model.eval()
        test_loss = 0.
        for batch_idx, (x,_) in enumerate(test_loader):
            x = x.to(device)
            # if args.smooth:
            noisy_x = x + torch.randn_like(x) * noise
            clean_x = x + torch.randn_like(x) * clean_noise
            x = torch.cat([noisy_x, clean_x], dim=2)
            x = Variable(x).to(device)
            output = model(x)[:, :, x.size()[-1]:, :]
            loss = loss_op(clean_x, output)
            test_loss += loss.item()
            del loss, output

        deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
        print("Epoch: {} - test loss : {}".format(epoch, test_loss / deno))
        
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(model_dir, '{}_{}.pth'.format(model_name, epoch))
            torch.save(model.state_dict(), ckpt_path)

            # Save some debug images
            debug_images(sample_model, img_dir, epoch=epoch)
        del sample_model

def run_training():
    print("Stage 2 training")
    train_loader, test_loader = build_dataloaders()
    train(model, train_loader, test_loader)


def run_conditional_generation(stage1_path):
    """
    Load images from stage 1 to generate images from stage 2
    """
    stage1_images = torch.load(stage1_path, map_location=device)
    
    stage2_images = sample(model, stage1_images)
    torch.save(stage2_images, "images/stage2_images.pth")

    # Hereafter is for visualization purpose
    samples = rescaling_inv(stage2_images)
    f = plt.figure()
    grid_img = tutils.make_grid(samples.cpu(), nrow=1, padding=1)
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0))
    f.savefig("images/stage2_images.png", bbox_inches="tight")


def debug_images(sample_model, output_path, epoch):
    stage1_images = torch.load("images/stage1_images.pth", map_location=device)
    x = stage1_images[:5]
    x = torch.concat([x, x], dim=2)
    x = x.to(device)

    # Sampling from stage 2 using inverse CDF
    for i in range(-x.shape[-1], 0, 1):
        for j in range(x.shape[-1]):
            samples = sample_from_discretized_mix_logistic_inverse_CDF(
                x, model=sample_model, nr_mix=10, clamp=False, bisection_iter=20)
            x[:, :, i, j] = samples[:, :, i, j]

    img = rescaling_inv(x)[:, :, -x.shape[-1]:, :]
    img2 = rescaling_inv(x)[:, :, :-x.shape[-1], :]
    img_grid = tutils.make_grid(img, nrow=int(img.shape[0] ** 0.5), padding=0, pad_value=0)
    img2_grid = tutils.make_grid(img2, nrow=int(img.shape[0] ** 0.5), padding=0, pad_value=0)
    tutils.save_image(img_grid, os.path.join(output_path, f"img_{epoch}.png"))
    tutils.save_image(img2_grid, os.path.join(output_path, f"img2_{epoch}.png"))

def run_stage2_sampling():
    stage1_images = torch.load("images/stage1_images.pth", map_location=device)
    stage2_images = sample(model, stage1_images)
    torch.save(stage2_images, "images/stage2_images_128.pth")

    samples = rescaling_inv(stage2_images[:64])
    f = plt.figure()
    grid_img = tutils.make_grid(samples.cpu(), nrow=8, padding=0)
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0))
    f.savefig("images/stage2_samples.png", bbox_inches="tight")

if __name__ == "__main__":
    # run_training()
    # run_conditional_generation("images/stage1_images.pth")
    run_stage2_sampling()
