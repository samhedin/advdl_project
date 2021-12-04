import pdb;
import torch
from torch.autograd import Variable
from model import PixelCNN
from utils import sample_from_discretized_mix_logistic

sample_batch_size = 1
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

def main():
    print("Loading model from checkpoint...")
    device = torch.device("cuda")
    model = PixelCNN(nr_resnet=5, nr_filters=160, input_channels=3, nr_logistic_mix=10)
    ckpt = torch.load("models/pcnn_lr:0.00020_nr-resnet5_nr-filters160_2.pth", map_location="cpu")
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()

    print("Sampling...")
    sample_t = sample(model)
    sample_t = rescaling_inv(sample_t)   

if __name__ == "__main__":
    main()
