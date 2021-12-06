import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from src.pixelcnn.layers import *

# from layers import *
from src.pixelcnn.utils import *

# from utils import *
import numpy as np


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
        # x: [B, 3, 32, 32]
        # similar as done in the tf repo :
        if self.init_padding is None and not sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False)
            self.init_padding = padding.cuda() if x.is_cuda else padding

        if sample:
            xs = [int(y) for y in x.size()]
            padding = Variable(
                torch.ones(xs[0], 1, xs[2], xs[3]), requires_grad=False
            )  # [B, 1, 32, 32]
            padding = padding.cuda() if x.is_cuda else padding
            x = torch.cat((x, padding), 1)  # [B, 4, 32, 32]

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
        # u_list length 18
        # ul_list length 18

        ###    DOWN PASS    ###
        u = u_list.pop()  # [B, 160, 32, 32]
        ul = ul_list.pop()  # [B, 160, 32, 32]

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


def mix_logistic_loss(x, l, likelihood=False):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    x = x.permute(0, 2, 3, 1)
    l = l.permute(0, 2, 3, 1)
    xs = [int(y) for y in x.size()]
    ls = [int(y) for y in l.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = l[:, :, :, :nr_mix]
    l = (
        l[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    )  # 3 for mean, scale, coef
    means = l[:, :, :, :, :nr_mix]
    log_scales = torch.clamp(l[:, :, :, :, nr_mix : 2 * nr_mix], min=-7.0)

    coeffs = torch.tanh(l[:, :, :, :, 2 * nr_mix : 3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    x = x.contiguous()
    x = x.unsqueeze(-1) + (torch.zeros(xs + [nr_mix]).cuda()).detach()
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :] * x[:, :, :, 0, :]).view(
        xs[0], xs[1], xs[2], 1, nr_mix
    )

    m3 = (
        means[:, :, :, 2, :]
        + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :]
        + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :]
    ).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = x - means
    inv_stdv = torch.exp(-log_scales)
    mid_in = inv_stdv * centered_x
    log_probs = mid_in - log_scales - 2.0 * F.softplus(mid_in)

    if likelihood:
        log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)
        return log_sum_exp(log_probs)

    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    return -torch.sum(log_sum_exp(log_probs))


def binary_search(log_cdf, lb, ub, cdf_fun, n_iter=15):
    with torch.no_grad():
        for i in range(n_iter):
            mid = (lb + ub) / 2.
            mid_cdf_value = cdf_fun(mid)
            right_idxes = mid_cdf_value < log_cdf
            left_idxes = ~right_idxes
            lb[right_idxes] = torch.min(mid[right_idxes], ub[right_idxes])
            ub[left_idxes] = torch.max(mid[left_idxes], lb[left_idxes])

    return mid


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

    def log_cdf_pdf_g(values, mode='cdf', mixtures=False):
        values = values.unsqueeze(-1)
        centered_values = (values - means_g) / log_scales_g.exp()

        if mode == 'cdf':
            log_logistic_cdf = log_p_r_mixtures - log_p_r[..., None] - F.softplus(-centered_values)
            log_logistic_sf = log_p_r_mixtures - log_p_r[..., None] - F.softplus(centered_values)
            log_cdf = torch.logsumexp(log_softmax + log_logistic_cdf, dim=-1)
            log_sf = torch.logsumexp(log_softmax + log_logistic_sf, dim=-1)
            logit = log_cdf - log_sf

            return logit if not mixtures else (logit, log_logistic_cdf)

        elif mode == 'pdf':
            log_logistic_pdf = log_p_r_mixtures - log_p_r[..., None] - centered_values - log_scales_g - 2. * F.softplus(
                -centered_values)
            log_pdf = torch.logsumexp(log_softmax + log_logistic_pdf, dim=-1)

            return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

    x1 = binary_search(u_g, lbs.clone(), ubs.clone(), lambda x: log_cdf_pdf_g(x, mode='cdf'), bisection_iter)

    if len(noise) == 0:
        means_b = x1.unsqueeze(-1) * coeffs[:, :, :, 2, :] + x0.unsqueeze(-1) * coeffs[:, :, :, 1, :] + means[..., 2, :]
    else:
        means_b = (x1.unsqueeze(-1) + noise[:, :, :, 1].unsqueeze(-1)) * coeffs[:, :, :, 2, :] + \
                  (x0.unsqueeze(-1) + noise[:, :, :, 0].unsqueeze(-1)) * coeffs[:, :, :, 1, :] + means[..., 2, :]

    means_b = means_b.detach() #added, to make autograd sample correct
    log_scales_b = log_scales[..., 2, :]

    log_p_g, log_p_g_mixtures = log_cdf_pdf_g(x1, mode='pdf', mixtures=True)

    def log_cdf_pdf_b(values, mode='cdf', mixtures=False):
        values = values.unsqueeze(-1)
        centered_values = (values - means_b) / log_scales_b.exp()

        if mode == 'cdf':
            log_logistic_cdf = log_p_g_mixtures - log_p_g[..., None] - F.softplus(-centered_values)
            log_logistic_sf = log_p_g_mixtures - log_p_g[..., None] - F.softplus(centered_values)
            log_cdf = torch.logsumexp(log_softmax + log_logistic_cdf, dim=-1)
            log_sf = torch.logsumexp(log_softmax + log_logistic_sf, dim=-1)
            logit = log_cdf - log_sf

            return logit if not mixtures else (logit, log_logistic_cdf)

        elif mode == 'pdf':
            log_logistic_pdf = log_p_g_mixtures - log_p_g[..., None] - centered_values - log_scales_b - 2. * F.softplus(
                -centered_values)
            log_pdf = torch.logsumexp(log_softmax + log_logistic_pdf, dim=-1)

            return log_pdf if not mixtures else (log_pdf, log_logistic_pdf)

    x2 = binary_search(u_b, lbs.clone(), ubs.clone(), lambda x: log_cdf_pdf_b(x, mode='cdf'), bisection_iter)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2)
    return out


if __name__ == "__main__":
    """ testing loss with tf version """
    np.random.seed(1)
    xx_t = (np.random.rand(15, 32, 32, 100) * 3).astype("float32")
    yy_t = np.random.uniform(-1, 1, size=(15, 32, 32, 3)).astype("float32")
    x_t = Variable(torch.from_numpy(xx_t)).cuda()
    y_t = Variable(torch.from_numpy(yy_t)).cuda()
    loss = discretized_mix_logistic_loss(y_t, x_t)

    """ testing model and deconv dimensions """
    x = torch.cuda.FloatTensor(32, 3, 32, 32).uniform_(-1.0, 1.0)
    xv = Variable(x).cpu()
    ds = down_shifted_deconv2d(3, 40, stride=(2, 2))
    x_v = Variable(x)

    """ testing loss compatibility """
    model = PixelCNN(nr_resnet=3, nr_filters=100, input_channels=x.size(1))
    model = model.cuda()
    out = model(x_v)
    loss = discretized_mix_logistic_loss(x_v, out)
    print("loss : %s" % loss.data[0])
