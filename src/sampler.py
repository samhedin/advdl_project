from typing import Tuple

import torch
from torch import autograd

from src.pixelcnn.utils import mix_logistic_loss


def single_step_denoising(
    module, sample_batch_size: int = 1, noise: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample images from the smoothed distribution (p\tilde{x}). Implemented
    equation (2) in the paper
    """
    # First, sample to get x tilde
    x_tilde = module.sample(sample_batch_size).to(module.device)

    # Log PDF:
    logits = module.model(x_tilde).detach()
    # Require grad for x_tilde so we can get its gradients later
    x_tilde.requires_grad_(True)
    log_pdf = mix_logistic_loss(x_tilde, logits, likelihood=True)

    # Compute the gradient of the loss w.r.t x_tilde
    nabla = autograd.grad(log_pdf.sum(), x_tilde)[0]
    x_bar = x_tilde + noise ** 2 * nabla
    return x_bar, x_tilde
