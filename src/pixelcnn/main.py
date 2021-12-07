import time
import os
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
parser.add_argument('-t', '--save_interval', type=int, default=5, # Original: 10
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
args = parser.parse_args()

# reproducibility
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# assert not os.path.exists(os.path.join('runs', model_name)), '{} already exists!'.format(model_name)
# writer = SummaryWriter(log_dir=os.path.join('runs', model_name))

sample_batch_size = 5
obs = (1, 28, 28) if 'mnist' in args.dataset else (3, 32, 32)
input_channels = obs[0]
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5
kwargs = {'num_workers':4, 'pin_memory':True, 'drop_last':True}
noise = 0.3

model_name = 'pcnn_lr:{:.5f}_nr-resnet{}_nr-filters{}_noise-{}'.format(args.lr, args.nr_resnet, args.nr_filters, str(noise).replace(".", ""))


def smooth(image):
    """Smooth input image by adding gaussian noise and rescale its values betwen [-1, 1]"""
    image = image + torch.randn_like(image) * noise
    image = 2 * (image - image.min()) / (image.max() - image.min()) - 1
    return image


smooth_op = lambda image: smooth(image)
ds_transforms = transforms.Compose([transforms.ToTensor(), smooth_op])

if 'mnist' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, download=True, 
                        train=True, transform=ds_transforms), batch_size=args.batch_size, 
                            shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.MNIST(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    loss_op   = lambda real, fake : discretized_mix_logistic_loss_1d(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic_1d(x, args.nr_logistic_mix)

elif 'cifar' in args.dataset : 
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=True, 
        download=True, transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    test_loader  = torch.utils.data.DataLoader(datasets.CIFAR10(args.data_dir, train=False, 
                    transform=ds_transforms), batch_size=args.batch_size, shuffle=True, **kwargs)
    
    # loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake)
    sample_op = lambda x : sample_from_discretized_mix_logistic(x, args.nr_logistic_mix)
else:
    raise Exception('{} dataset not in {mnist, cifar10}'.format(args.dataset))

model = PixelCNN(nr_resnet=args.nr_resnet, nr_filters=args.nr_filters, 
            input_channels=input_channels, nr_logistic_mix=args.nr_logistic_mix)
model = model.cuda()

if args.load_params:
    load_part_of_model(model, args.load_params)
    # model.load_state_dict(torch.load(args.load_params))
    print('model parameters loaded')

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay)

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):
            with torch.no_grad():
                data_v = Variable(data)
                out   = model(data_v, sample=True)  # model.forward(data_v, sample=True)
                out_sample = sample_op(out)
                data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data

def single_step_denoising(model):
    device = torch.device("cuda")
    # First, sample to get x tilde
    x_tilde = sample(model)

    # Log PDF:
    logits = model(x_tilde).detach()  # logits don't require gradient
    xt_v = Variable(x_tilde, requires_grad=True).to(device)
    log_pdf = mix_logistic_loss(xt_v, logits, likelihood=True)

    nabla = autograd.grad(log_pdf.sum(), xt_v, create_graph=True)[0]
    x = x_tilde + noise**2 * nabla
    return x, x_tilde


def train():
    print('starting training')
    writes = 0
    for epoch in range(args.max_epochs):
        model.train(True)
        torch.cuda.synchronize()
        train_loss = 0.
        time_ = time.time()
        model.train()
        for batch_idx, (input,_) in enumerate(train_loader):
            input = input.cuda(non_blocking=True)
            input = Variable(input)
            output = model(input)
            loss = loss_op(input, output)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            if (batch_idx +1) % args.print_every == 0 : 
                deno = args.print_every * args.batch_size * np.prod(obs) * np.log(2.)
                # writer.add_scalar('train/bpd', (train_loss / deno), writes)
                print('loss : {:.4f}, time : {:.4f}'.format(
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
            input_var = Variable(input)
            output = model(input_var)
            loss = loss_op(input_var, output)
            test_loss += loss.item()
            del loss, output

        deno = batch_idx * args.batch_size * np.prod(obs) * np.log(2.)
        # writer.add_scalar('test/bpd', (test_loss / deno), writes)
        print('test loss : %s' % (test_loss / deno))
        
        if (epoch + 1) % args.save_interval == 0:
            torch.save(model.state_dict(), 'models/{}_{}.pth'.format(model_name, epoch))
            print('sampling...')
            sample_t = sample(model)
            sample_t = rescaling_inv(sample_t)
            tutils.save_image(sample_t,'images/{}_{}.png'.format(model_name, epoch), 
                    nrow=5, padding=0)

        # print("Single-step denoising")
        # x, x_tilde = single_step_denoising(model)
        # x = rescaling_inv(x)
        # x_tilde = rescaling_inv(x_tilde)

        # f = plt.figure()
        # a = f.add_subplot(2, 1, 1)
        # a.title.set_text("Before denoising")

        # grid_img = tutils.make_grid(x_tilde.cpu())
        # plt.imshow(grid_img.permute(1, 2, 0))

        # a = f.add_subplot(2, 1, 2)
        # a.title.set_text("After single step denoising")
        # grid_img = tutils.make_grid(x.cpu())
        # plt.imshow(grid_img.permute(1, 2, 0))
        # plt.savefig("images/ssd_{}_{}.png".format(model_name, epoch))

if __name__ == "__main__":
    # train()
    print("Single-step denoising")
    x, x_tilde = single_step_denoising(model)
    x = rescaling_inv(x)
    x_tilde = rescaling_inv(x_tilde)

    f = plt.figure()
    a = f.add_subplot(2, 1, 1)
    a.title.set_text("Before denoising")

    grid_img = tutils.make_grid(x_tilde.cpu())
    plt.imshow(grid_img.permute(1, 2, 0))

    a = f.add_subplot(2, 1, 2)
    a.title.set_text("After single step denoising")
    grid_img = tutils.make_grid(x.cpu())
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.savefig("images/ssd_{}_{}.png".format(model_name, "after_9"))