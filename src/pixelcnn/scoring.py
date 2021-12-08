import time
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data

from torchvision.models.inception import inception_v3

import numpy as np
from scipy.stats import entropy

from model import PixelCNN
from main import single_step_denoising, rescaling_inv, sample
from utils import load_part_of_model

torch.manual_seed(1)
np.random.seed(1)


class IgnoreLabelDataset(torch.utils.data.Dataset):
        def __init__(self, orig):
            self.orig = orig

        def __getitem__(self, index):
            return self.orig[index][0]

        def __len__(self):
            return len(self.orig)


def inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def compute_model_inception_score(model_path=None, sample_batch_size=100, batch_size=64):
    device = torch.device("cuda")
    # load model
    print(f"Loading the model from {model_path}")
    model = PixelCNN(nr_resnet=5, nr_filters=160, nr_logistic_mix=10, input_channels=3)
    load_part_of_model(model, model_path)
    model.eval()
    model.to(device)

    # sample images
    print("Single-step denoising from model...")
    ssd_start = time.time()
    x_bar, _ = single_step_denoising(model, sample_batch_size=sample_batch_size)
    # x_bar = sample(model, sample_batch_size=sample_batch_size)
    ssd_end = time.time()
    print(f"Time for SSD: {ssd_end - ssd_start}")

    x_bar = rescaling_inv(x_bar)  # [B, 3, 32, 32]
    print("x_bar", x_bar.shape)

    print("Computing inception score...")
    img_dataset = IgnoreLabelDataset(torch.utils.data.TensorDataset(x_bar))
    is_mean, is_std = inception_score(img_dataset, cuda=True, batch_size=batch_size, resize=True)
    print(is_mean, is_std)


if __name__ == '__main__':
    cfg = {
        "model_path": "models/exp3b/pcnn_lr:0.00020_nr-resnet5_nr-filters160_noise-03_99.pth",
        "sample_batch_size": 64 * 2,
        "batch_size": 64
    }
    print("Computing Inception score with settings...")
    print(cfg)
    start = time.time()
    compute_model_inception_score(**cfg)
    end = time.time()
    total_t = end - start
    print(f"Time spent: {total_t}")
