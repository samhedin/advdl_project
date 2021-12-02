from typing import Any, Optional

import torch
from torch import autograd
from torch.autograd import Variable

from src.utils import (
    discretized_mix_logistic_loss,
    sample_from_discretized_mix_logistic
)


def single_step_denoising(model, sample_batch_size: int = 1, nr_logistic_mix: int = 10, sigma: float = 0.1, obs: Optional[Any] = (3, 32, 32)):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    sample_op = lambda x : sample_from_discretized_mix_logistic(x, nr_logistic_mix)
    model.train(False)
    model.to(device)

    # First, sample to get x tilde
    x_tilde = torch.zeros(sample_batch_size, obs[0], obs[1], obs[2]).to(device)

    for i in range(obs[1]):
        for j in range(obs[2]):
            with torch.no_grad():
                data_v = Variable(x_tilde).to(device)
                out = model(data_v, sample=True)
                out_sample = sample_op(out)
                x_tilde[:, :, i, j] = out_sample.data[:, :, i, j]
    
    # Log PDF:
    xt_v = Variable(x_tilde, requires_grad=True).to(device)
    logits = model(xt_v, sample=True)
    log_pdf = discretized_mix_logistic_loss(xt_v, logits)

    nabla = autograd.grad(log_pdf, xt_v)[0]
    x = x_tilde + sigma**2 * nabla
    return x
