from typing import Tuple

import torch
from torch import autograd

from src.pixelcnn.utils import mix_logistic_loss


def single_step_denoising(
    module, sample_batch_size: int = 1, noise: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample images from the smoothed distribution (p\tilde{x}).
    Implemente equation (2) in the paper
    """
    # First, sample to get x tilde, i.e., noisy sample
    x_tilde = module.sample(sample_batch_size).to(module.device)

    # Log PDF:
    x_t = autograd.Variable(x_tilde, requires_grad=True).to(module.device)
    logits = module.model(x_t).detach()
    # Require grad for x_tilde so we can get its gradients later
    # x_tilde.requires_grad_(True)
    log_pdf = mix_logistic_loss(x_t, logits, likelihood=True)
    # Compute the gradient of the loss w.r.t x_tilde
    nabla = autograd.grad(log_pdf.sum(), x_t, create_graph=True)[0]
    x_bar = x_tilde + (noise ** 2) * nabla
    x_bar = x_bar.detach().data
    x_tilde = x_tilde.detach().data
    return x_bar, x_tilde
