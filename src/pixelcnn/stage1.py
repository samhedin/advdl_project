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
from tensorboardX import SummaryWriter
from torchvision import utils as tutils
from utils import * 
from model import * 
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    parser.add_argument('-b', '--batch_size', type=int, default=64,
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

sample_batch_size = 64
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':4, 'pin_memory':True, 'drop_last':True}
noise = 0.3

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}_noise-{}'.format(args.lr, args.nr_resnet, args.nr_filters, str(noise).replace(".", ""))

# Recale the image to range [-1, 1]
ds_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
    download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)

# loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)

print("Creating model...")
model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()

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


def sample(model, sample_batch_size=5):
    model.train(False)

    # num_batches = sample_batch_size // 64 if sample_batch_size >= 64 else 1
    num_batches = sample_batch_size // 1500 if sample_batch_size > 1500 else 1

    sample_op = lambda x : sample_from_discretized_mix_logistic(x, 10)

    out_data = []
    for bid in range(num_batches):
        start_time = time.time()
        # batch_size = 64 if num_batches > 1 else sample_batch_size
        batch_size = 1500 if num_batches > 1 else sample_batch_size
        data = torch.zeros(batch_size, obs[0], obs[1], obs[2])
        data = data.cuda()
        print(f"Batch {bid}, batch_size={batch_size}")
        for i in range(obs[1]):
            for j in range(obs[2]):
                with torch.no_grad():
                    data_v = Variable(data).cuda()
                    out   = model(data_v, sample=True)  # model.forward(data_v, sample=True)
                    out_sample = sample_op(out)
                    data[:, :, i, j] = out_sample.data[:, :, i, j]
        out_data.append(data)
        end_time = time.time()
        batch_time = end_time - start_time
        print(f"Finish generating for batch {bid} / {num_batches} batches; time: {batch_time}")

    if len(out_data) == 1:
        return out_data[0]

    return torch.concat(out_data)

def single_step_denoising(model, sample_batch_size=None, sampling=True, x_tildes=None):
    model.cuda()
    device = torch.device("cuda")
    print("sampling", sampling)
    if sampling is True:
        # First, sample to get x tilde
        x_tildes = sample(model, sample_batch_size=sample_batch_size) # [B, 3, 32, 32]
    elif sampling is False and x_tildes is None:
        raise RuntimeError("When sampling is False, x_tildes must be provided")
    x_tildes = torch.split(x_tildes, 64) #

    # Log PDF:
    x_bar, xt_acc = [], []
    for x_tilde in x_tildes:
        if x_tilde.shape[0] != 64:
            print("Hacky, avoid last batch")
            continue
        x_tilde = x_tilde.to(device)
        logits = model(x_tilde, sample=False).detach()  # logits don't require gradient
        xt_v = Variable(x_tilde, requires_grad=True).to(device)
        log_pdf = mix_logistic_loss(xt_v, logits, likelihood=True)

        nabla = autograd.grad(log_pdf.sum(), xt_v, create_graph=True)[0]
        x = x_tilde + noise**2 * nabla
        x_bar.append(x)
        xt_acc.append(x_tilde)
    
    return torch.concat(x_bar, dim=0), torch.concat(xt_acc, dim=0)

def train():
    print('starting training')
    writes = 0
    for epoch in range(args.max_epochs):
        if args.resume_from:
            epoch += args.resume_from
        model.train(True)
        torch.cuda.synchronize()
        train_loss = 0.
        time_ = time.time()
        model.train()
        for batch_idx, (input,_) in enumerate(train_loader):
            input = input.cuda(non_blocking=True)
            if args.smooth:
                input = input + torch.randn_like(input) * noise
            input = Variable(input)
            output = model(input)
            loss = loss_op(input, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx +1) % args.print_every == 0 : 
                deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
                print('Epoch: {} - loss : {:.4f}, time : {:.4f}'.format(
                    epoch,
                    (train_loss / deno), 
                    (time.time() - time_)))
                train_loss = 0.
                writes += 1
                time_ = time.time()    

        # decrease learning rate
        scheduler.step()
        
        torch.cuda.synchronize()
        model.eval()
        test_loss = 0.
        for batch_idx, (input,_) in enumerate(test_loader):
            input = input.cuda(non_blocking=True)
            if args.smooth:
                input = input + torch.randn_like(input) * noise
            input_var = Variable(input)
            output = model(input_var)
            loss = loss_op(input_var, output)
            test_loss += loss.item()
            del loss, output

        deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
        print("Epoch: {} - test loss : {}".format(epoch, test_loss / deno))
        
        if (epoch + 1) % args.save_interval == 0:
            ckpt_path = os.path.join(model_dir, '{}_{}.pth'.format(model_name, epoch))
            torch.save(model.state_dict(), ckpt_path)
            print('sampling...')
            sample_t = sample(model)
            sample_t = rescaling_inv(sample_t)

            img_path = os.path.join(img_dir, '{}_{}.png'.format(model_name, epoch))
            nrow = 1 if sample_batch_size <= 8 else sample_batch_size // 8
            tutils.save_image(sample_t, img_path, nrow=nrow, padding=0)

def run_single_step_denoising():
    print("Single-step denoising")
    x, x_tilde = single_step_denoising(model, sample_batch_size=sample_batch_size)
    x = rescaling_inv(x)
    x_tilde = rescaling_inv(x_tilde)

    nrow = 1
    if (sample_batch_size // 8) > 1:
        nrow = sample_batch_size // 8
    f = plt.figure()
    grid_img = tutils.make_grid(x.cpu(), nrow=nrow, padding=0)
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0))
    f.savefig("images/exp2a/exp2a_ssd.png", bbox_inches="tight")

def run_sampling():
    samples = sample(model, sample_batch_size=sample_batch_size)
    samples = rescaling_inv(samples)

    nrow = 1
    if (sample_batch_size // 8) > 1:
        nrow = sample_batch_size // 8
    f = plt.figure()
    grid_img = tutils.make_grid(samples.cpu(), nrow=nrow, padding=0)
    plt.axis("off")
    plt.imshow(grid_img.permute(1, 2, 0))
    f.savefig("images/exp2b/exp2b_baseline.png", bbox_inches="tight")

def experiment_2a():
    print("Experiment 2a")
    print(args)
    train()

def sampling_for_stage2():
    print(f"Generating {5*1500} samples from stage 1 for stage 2")
    samples = sample(model, sample_batch_size=5*1500)
    torch.save(samples, "images/stage1_images_7.5k.pth")

def run_ssd_for_final_report():
    print(f"Generating {5*1500} samples from SSD for final report")
    x_tidle = single_step_denoising(model, sample_batch_size=5*1500)
    x_tidle = rescaling_inv(x_tidle)

    torch.save(x_tidle, "images/stage1_ssd_7.5k.pth")

def run_sampling_baseline_for_report():
    print(f"Generating {5*1500} baseline samples for final report")
    samples = sample(model, sample_batch_size=5*1500)
    torch.save(samples, "images/stage1_baseline_7.5k.pth")

if __name__ == "__main__":
    # train()
    # run_single_step_denoising()
    # run_sampling()
    # experiment_2a()
    # sampling_for_stage2()
    # run_ssd_for_final_report()
    run_sampling_baseline_for_report()
