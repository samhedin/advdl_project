from typing import Any, Optional

import torch
from torch import autograd
from torch.autograd import Variable

from src.utils import discretized_mix_logistic_loss


def single_step_denoising(module, sample_batch_size: int = 1, sigma: float = 0.1):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # First, sample to get x tilde
    x_tilde = module.sample(sample_batch_size)

    # Log PDF:
    xt_v = Variable(x_tilde, requires_grad=True).to(device)
    module.model.to(device)
    logits = module.model(xt_v, sample=True)
    log_pdf = discretized_mix_logistic_loss(xt_v, logits)

    nabla = autograd.grad(log_pdf, xt_v)[0]
    x = x_tilde + sigma**2 * nabla
    return x
