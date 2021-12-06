import pdb;
import torch
import torch.autograd as autograd
from torch.autograd import Variable
from model import PixelCNN, mix_logistic_loss
from utils import sample_from_discretized_mix_logistic

from torchvision import utils as tvutils
import matplotlib.pyplot as plt

sample_batch_size = 2
noise = 0.3
obs = (3, 32, 32)
sample_op = lambda x : sample_from_discretized_mix_logistic(x, 10)
rescaling     = lambda x : (x - .5) * 2.
rescaling_inv = lambda x : .5 * x  + .5

def sample(model):
    model.train(False)
    data = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2])
    data = data.cuda()
    for i in range(obs[1]):
        for j in range(obs[2]):
            with torch.no_grad():
                data_v = Variable(data)
                out   = model(data_v, sample=True) # [B, 100, 32, 32]
                out_sample = sample_op(out) # [B, 3, 32, 32]
                data[:, :, i, j] = out_sample.data[:, :, i, j]
    return data


def single_step_denoising(model, sigma: float = 0.3):
    device = torch.device("cuda")
    # First, sample to get x tilde
    x_tilde = sample(model).to(device)

    # Log PDF:
    xt_v = Variable(x_tilde, requires_grad=True).to(device)
    logits = model(xt_v, sample=True)
    log_pdf = mix_logistic_loss(xt_v, logits, likelihood=True)

    nabla = autograd.grad(log_pdf.sum(), xt_v, create_graph=True)[0]
    x = x_tilde + sigma**2 * nabla
    return x, x_tilde


def rescale_image(x_in: torch.Tensor) -> torch.Tensor:
    """
    Rescale the input tensor to range [0, 1]. This function should be used to rescale
    the samples from model sampling or SSD sampling process before visualizing or saving images
    to files. To rescale the tensor for other purposes, use other functions
    TODO: This function is currently working for x_bar, i.e., samples from single-step denoising
    Args:
        x_in - torch.Tensor: of dim [B, C, H, W]
    
    Returns:
        x - torch.Tensor of dim B, C, H, W
    """
    B, C, H, W = x_in.size()
    x = x_in.view(B, C, -1)
    x -= x.min(1, keepdim=True)[0]
    x /= x.max(1, keepdim=True)[0]
    return x.view(B, C, H, W)

def main():
    print("Loading model from checkpoint...")
    device = torch.device("cuda")
    model = PixelCNN(nr_resnet=5, nr_filters=160, input_channels=3, nr_logistic_mix=10)
    ckpt = torch.load("models/pcnn_lr:0.00020_nr-resnet5_nr-filters160_1.pth", map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    x, x_tilde = single_step_denoising(model, sigma=noise)

    x_tilde = rescaling_inv(x_tilde)
    grid_xt = tvutils.make_grid(x_tilde, nrow=1)
    plt.imsave("grid_xt.png", grid_xt.permute(1, 2, 0).cpu().numpy())

    x = rescale_image(x)
    grid_x = tvutils.make_grid(x, nrow=1)
    plt.imsave("grid_x.png", grid_x.permute(1, 2, 0).cpu().numpy())

if __name__ == "__main__":
    main()
