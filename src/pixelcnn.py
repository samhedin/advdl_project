#!/usr/bin/env python3
import src.utils as utils
import os

from torchvision.utils import save_image
from tensorboardX import SummaryWriter
import time
from torch.optim import lr_scheduler
import torch.optim as optim
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.layers import *
from src.utils import *
import numpy as np


class CNN_helper():
    def __init__(self, args, train_loader, test_loader, pretrained=False, stage=1):
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.args = args
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.obs = (3, 32, 32)

        self.model_name = f'pcnn_lr_{self.args.lr:.5f}_nr-resnet_{self.args.nr_resnet}_nr-filters_{self.args.nr_filters}_stage_{stage}'#
        # Due to memory constraint, we max out at 3 Resnet, the paper has 5
        self.model = PixelCNN(nr_resnet=3, nr_filters=160)
        self.model.to(self.device)
        if pretrained:
            pixelcnnpp_pretrained = "pretrained/pixel-cnn-pp/pcnn_lr.0.00040_nr-resnet5_nr-filters160_889.pth"
            utils.load_part_of_model(self.model, pixelcnnpp_pretrained)

    def train(self):
        rescaling_inv = lambda x : .5 * x  + .5
        writer = SummaryWriter(log_dir=os.path.join('runs', self.model_name))
        loss_op   = lambda real, fake : discretized_mix_logistic_loss(real, fake, self.args)
        optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=self.args.lr_decay)

        print('starting training')
        writes = 0
        for epoch in range(self.args.max_epochs):
            print(f"epoch: {epoch}")
            self.model.train(True)
            if self.args.cuda == 1:
                torch.cuda.synchronize()
            train_loss = 0.
            time_ = time.time()
            self.model.train()
            for batch_idx, (x,_) in enumerate(self.train_loader):
                if self.args.cuda == 1:
                    x = x.cuda(non_blocking=True)
                x = Variable(x)
                output = self.model(x)
                loss = loss_op(x, output)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                if (batch_idx +1) % self.args.print_every == 0 :
                    deno = self.args.print_every * self.args.batch_size * np.prod(self.obs) * np.log(2.)
                    writer.add_scalar('train/bpd', (train_loss / deno), writes)
                    print('loss : {:.4f}, time : {:.4f}'.format(
                        (train_loss / deno),
                        (time.time() - time_)))
                    train_loss = 0.
                    writes += 1
                    time_ = time.time()

            # decrease learning rate
            scheduler.step()

            if self.args.cuda == 1:
                torch.cuda.synchronize()
            self.model.eval()
            test_loss = 0.
            for batch_idx, (x,_) in enumerate(self.test_loader):
                if self.args.cuda == 1:
                    x = x.cuda(non_blocking=True)
                input_var = Variable(x)
                output = self.model(input_var)
                loss = loss_op(input_var, output)
                test_loss += loss.item()
                del loss, output

                deno = batch_idx * self.args.batch_size * np.prod(self.obs) * np.log(2.)
                writer.add_scalar('test/bpd', (test_loss / deno), writes)
                print('test loss : %s' % (test_loss / deno))

            if (epoch + 1) % self.args.save_interval == 0:
                print("saving image")
                torch.save(self.model.state_dict(), f'models/{self.model_name}_{epoch}.pth')
                print('sampling...')
                sample_t = self.sample()
                sample_t = rescaling_inv(sample_t)
                save_image(sample_t,'imgs/{}_{}.png'.format(self.model_name, epoch),
                        nrow=5, padding=0)

    def sample(self, sample_batch_size=1):
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

        sample_op = lambda x : sample_from_discretized_mix_logistic(x, self.args.nr_logistic_mix)
        self.model.train(False)
        self.model.to(device)

        data = torch.zeros(sample_batch_size, self.obs[0], self.obs[1], self.obs[2])
        data.to(device)
        for i in range(self.obs[1]):
            for j in range(self.obs[2]):
                with torch.no_grad():
                    data_v = Variable(data).to(device)
                    out   = self.model(data_v, sample=True)
                    out_sample = sample_op(out)
                    data[:, :, i, j] = out_sample.data[:, :, i, j]
        return data


class PixelCNNLayer_up(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_up, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList(
            [
                gated_resnet(
                    nr_filters,
                    down_shifted_conv2d,
                    resnet_nonlinearity,
                    skip_connection=0,
                )
                for _ in range(nr_resnet)
            ]
        )

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList(
            [
                gated_resnet(
                    nr_filters,
                    down_right_shifted_conv2d,
                    resnet_nonlinearity,
                    skip_connection=1,
                )
                for _ in range(nr_resnet)
            ]
        )

    def forward(self, u, ul):
        u_list, ul_list = [], []

        for i in range(self.nr_resnet):
            u = self.u_stream[i](u)
            ul = self.ul_stream[i](ul, a=u)
            u_list += [u]
            ul_list += [ul]

        return u_list, ul_list


class PixelCNNLayer_down(nn.Module):
    def __init__(self, nr_resnet, nr_filters, resnet_nonlinearity):
        super(PixelCNNLayer_down, self).__init__()
        self.nr_resnet = nr_resnet
        # stream from pixels above
        self.u_stream = nn.ModuleList(
            [
                gated_resnet(
                    nr_filters,
                    down_shifted_conv2d,
                    resnet_nonlinearity,
                    skip_connection=1,
                )
                for _ in range(nr_resnet)
            ]
        )

        # stream from pixels above and to thes left
        self.ul_stream = nn.ModuleList(
            [
                gated_resnet(
                    nr_filters,
                    down_right_shifted_conv2d,
                    resnet_nonlinearity,
                    skip_connection=2,
                )
                for _ in range(nr_resnet)
            ]
        )

    def forward(self, u, ul, u_list, ul_list):
        for i in range(self.nr_resnet):
            u = self.u_stream[i](u, a=u_list.pop())
            ul = self.ul_stream[i](ul, a=torch.cat((u, ul_list.pop()), 1))

        return u, ul


class PixelCNN(nn.Module):
    def __init__(
        self,
        nr_resnet=5,
        nr_filters=80,
        nr_logistic_mix=10,
        resnet_nonlinearity="concat_elu",
        input_channels=3,
    ):
        super(PixelCNN, self).__init__()
        if resnet_nonlinearity == "concat_elu":
            self.resnet_nonlinearity = lambda x: concat_elu(x)
        else:
            raise Exception(
                "right now only concat elu is supported as resnet nonlinearity."
            )

        self.nr_filters = nr_filters
        self.input_channels = input_channels
        self.nr_logistic_mix = nr_logistic_mix
        self.right_shift_pad = nn.ZeroPad2d((1, 0, 0, 0))
        self.down_shift_pad = nn.ZeroPad2d((0, 0, 1, 0))

        down_nr_resnet = [nr_resnet] + [nr_resnet + 1] * 2
        self.down_layers = nn.ModuleList(
            [
                PixelCNNLayer_down(
                    down_nr_resnet[i], nr_filters, self.resnet_nonlinearity
                )
                for i in range(3)
            ]
        )

        self.up_layers = nn.ModuleList(
            [
                PixelCNNLayer_up(nr_resnet, nr_filters, self.resnet_nonlinearity)
                for _ in range(3)
            ]
        )

        self.downsize_u_stream = nn.ModuleList(
            [
                down_shifted_conv2d(nr_filters, nr_filters, stride=(2, 2))
                for _ in range(2)
            ]
        )

        self.downsize_ul_stream = nn.ModuleList(
            [
                down_right_shifted_conv2d(nr_filters, nr_filters, stride=(2, 2))
                for _ in range(2)
            ]
        )

        self.upsize_u_stream = nn.ModuleList(
            [
                down_shifted_deconv2d(nr_filters, nr_filters, stride=(2, 2))
                for _ in range(2)
            ]
        )

        self.upsize_ul_stream = nn.ModuleList(
            [
                down_right_shifted_deconv2d(nr_filters, nr_filters, stride=(2, 2))
                for _ in range(2)
            ]
        )

        self.u_init = down_shifted_conv2d(
            input_channels + 1, nr_filters, filter_size=(2, 3), shift_output_down=True
        )

        self.ul_init = nn.ModuleList(
            [
                down_shifted_conv2d(
                    input_channels + 1,
                    nr_filters,
                    filter_size=(1, 3),
                    shift_output_down=True,
                ),
                down_right_shifted_conv2d(
                    input_channels + 1,
                    nr_filters,
                    filter_size=(2, 1),
                    shift_output_right=True,
                ),
            ]
        )

        num_mix = 3 if self.input_channels == 1 else 10
        self.nin_out = nin(nr_filters, num_mix * nr_logistic_mix)
        self.init_padding = None

    def forward(self, x, sample=False):
        # similar as done in the tf repo :
        # if self.init_padding is None and not sample: # TODO: remove True after debugging
        if not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)

        ###      UP PASS    ###
        x = x if sample else torch.cat((x, self.init_padding), 1)
        u_list = [self.u_init(x)]
        ul_list = [self.ul_init[0](x) + self.ul_init[1](x)]
        for i in range(3):
            # resnet block
            u_out, ul_out = self.up_layers[i](u_list[-1], ul_list[-1])
            u_list += u_out
            ul_list += ul_out

            if i != 2:
                # downscale (only twice)
                u_list += [self.downsize_u_stream[i](u_list[-1])]
                ul_list += [self.downsize_ul_stream[i](ul_list[-1])]

        ###    DOWN PASS    ###
        u = u_list.pop()
        ul = ul_list.pop()

        for i in range(3):
            # resnet block
            u, ul = self.down_layers[i](u, ul, u_list, ul_list)

            # upscale (only twice)
            if i != 2:
                u = self.upsize_u_stream[i](u)
                ul = self.upsize_ul_stream[i](ul)

        x_out = self.nin_out(F.elu(ul))

        assert len(u_list) == len(ul_list) == 0, pdb.set_trace()

        return x_out

def sample_from_discretized_mix_logistic_inverse_CDF(x, model, nr_mix, noise=[], u=None, clamp=True, bisection_iter=15, T=1):
    # Pytorch ordering
    l = model(x)
    l = l.permute(0, 2, 3, 1)
    ls = [int(y) for y in l.size()]
    xs = ls[:-1] + [3]

    #added
    if len(noise) != 0:
        noise = noise.permute(0, 2, 3, 1)
    #added

    # unpack parameters
    logit_probs = l[:, :, :, :nr_mix] / T
    l = l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    if u is None:
        u = l.new_empty(l.shape[0], l.shape[1] * l.shape[2] * 3)
        u.uniform_(1e-5, 1. - 1e-5)
        u = torch.log(u) - torch.log(1. - u)

    u_r, u_g, u_b = torch.chunk(u, chunks=3, dim=-1)

    u_r = u_r.reshape(ls[:-1])
    u_g = u_g.reshape(ls[:-1])
    u_b = u_b.reshape(ls[:-1])

    log_softmax = torch.log_softmax(logit_probs, dim=-1)
    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix: 3 * nr_mix])
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix:2 * nr_mix], min=-7.) + np.log(T)
    if clamp:
        ubs = l.new_ones(ls[:-1])
        lbs = -ubs
    else:
        ubs = l.new_ones(ls[:-1]) * 20.
        lbs = -ubs

    means_r = means[..., 0, :]
    log_scales_r = log_scales[..., 0, :]

    def log_cdf_pdf_r(values, mode='cdf', mixtures=False):
        values = values.unsqueeze(-1)
        centered_values = (values - means_r) / log_scales_r.exp()

        if mode == 'cdf':
            log_logistic_cdf = -F.softplus(-centered_values)
            log_logistic_sf = -F.softplus(centered_values)
            log_cdf = torch.logsumexp(log_softmax + log_logistic_cdf, dim=-1)
            log_sf = torch.logsumexp(log_softmax + log_logistic_sf, dim=-1)
            logit = log_cdf - log_sf

            return logit if not mixtures else (logit, log_logistic_cdf)

        elif mode == 'pdf':
            log_logistic_pdf = -centered_values - log_scales_r - 2. * F.softplus(-centered_values)
            log_pdf = torch.logsumexp(log_softmax + log_logistic_pdf, dim=-1)

            return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

    x0 = binary_search(u_r, lbs.clone(), ubs.clone(), lambda x: log_cdf_pdf_r(x, mode='cdf'), bisection_iter)

    if len(noise) == 0:
        means_g = x0.unsqueeze(-1) * coeffs[:, :, :, 0, :] + means[..., 1, :]
    else:
        means_g = (x0.unsqueeze(-1) + noise[:, :, :, 0].unsqueeze(-1)) * coeffs[:, :, :, 0, :] + means[..., 1, :]

    means_g = means_g.detach() #added, to make autograd sample correct
    log_scales_g = log_scales[..., 1, :]

    log_p_r, log_p_r_mixtures = log_cdf_pdf_r(x0, mode='pdf', mixtures=True)


def mix_logistic_loss(x: torch.Tensor, logits: torch.Tensor):
    # TODO: Complete this function
    pass


# if __name__ == "__main__":
#     """testing loss with tf version"""
#     np.random.seed(1)
# xx_t = (np.random.rand(15, 32, 32, 100) * 3).astype("float32")
#     yy_t = np.random.uniform(-1, 1, size=(15, 32, 32, 3)).astype("float32")
#     x_t = Variable(torch.from_numpy(xx_t)).cuda()
#     y_t = Variable(torch.from_numpy(yy_t)).cuda()
#     loss = discretized_mix_logistic_loss(y_t, x_t)

#     """ testing self.model and deconv dimensions """
#     x = torch.cuda.FloatTensor(32, 3, 32, 32).uniform_(-1.0, 1.0)
#     xv = Variable(x).cpu()
#     ds = down_shifted_deconv2d(3, 40, stride=(2, 2))
#     x_v = Variable(x)

#     """ testing loss compatibility """
#     self.model = PixelCNN(nr_resnet=3, nr_filters=100, input_channels=x.size(1))
#     self.model = self.model.cuda()
#     out = self.model(x_v)
#     loss = discretized_mix_logistic_loss(x_v, out)
#     print("loss : %s" % loss.data[0])
