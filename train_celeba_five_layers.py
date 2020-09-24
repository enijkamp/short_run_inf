import os
import sys
import datetime
import math
import shutil
import random
import argparse
import logging
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

import pygrid

import math

cuda = torch.cuda.is_available()

########################################################################################################

def parse_args():
    args = argparse.Namespace()

    args.seed = 1
    args.device = 0

    args.n_epochs = 600

    args.img_sz = 32
    args.img_ch = 3
    args.n_batch = 100

    args.lr = 8e-5
    args.lr_decay = 7
    args.beta1 = 0.5

    args.use_bn = False
    args.use_ladder = False
    args.use_skip = True

    args.z_dims = [64, 64, 64, 96, 128]

    args.z_sigma = 0.15
    args.z_sigma_decay = -1
    args.z_step_size = 0.05
    args.z_n_iters = 25
    args.z_with_noise = True

    return args


def create_args_grid():
    # TODO(nijkamp): add your enumeration of parameters here

    seed = [2]
    z_step_size = [0.05]
    z_sigma = [0.15]
    z_sigma_decay = [-1]
    lr = [8e-5]
    lr_decay = [7]
    beta1 = [0.5]
    z_n_iters = [25]

    args_list = [seed, z_step_size, z_sigma, z_sigma_decay, lr, lr_decay, beta1, z_n_iters]

    opt_list = []
    for i, args in enumerate(itertools.product(*args_list)):
        opt_job = {'job_id': int(i), 'status': 'open'}
        opt_args = {
            'seed': args[0],
            'z_step_size': args[1],
            'z_sigma': args[2],
            'z_sigma_decay': args[3],
            'lr': args[4],
            'lr_decay': args[5],
            'beta1': args[6],
            'z_n_iters': args[7],
        }
        # TODO(nijkamp): add your result metric here
        opt_result = {'epoch': 0, 'i': 0, 'mse': 0.0, 'mse_best': 0.0, 'fid': 0.0, 'fid_best': 0.0, 'success': 0}

        # opt_list += [{**opt_job, **opt_args, **opt_result}]
        opt_list += [merge_dicts(opt_job, opt_args, opt_result)]

    return opt_list


def update_job_result(job_opt, job_stats):
    # TODO add your result metric here
    job_opt['epoch'] = job_stats['epoch']
    job_opt['i'] = job_stats['i']
    job_opt['mse'] = job_stats['mse']
    job_opt['fid'] = job_stats['fid']
    job_opt['mse_best'] = job_stats['mse_best']
    job_opt['fid_best'] = job_stats['fid_best']
    job_opt['success'] = job_stats['success']

########################################################################################################

def gelu(x):
    if hasattr(torch.nn.functional, 'gelu'):
        return torch.nn.functional.gelu(x.float())
    else:
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

########################################################################################################

def stochastic_gaussian(input_mean, input_var, is_log_var, eps=None):
    input_var = torch.exp(0.5 * input_var) if is_log_var else torch.sqrt(input_var)
    outputs = input_mean + input_var * eps
    return outputs


class LadderStochastic(nn.Module):
    def __init__(self, feature_dim, z_dim, is_log_var=True, use_bn=False, eps=1e-8, activation=gelu):
        super(LadderStochastic, self).__init__()
        self.use_bn = use_bn
        self.activation = activation

        self.mean_conv = nn.Conv2d(feature_dim, z_dim, kernel_size=5, stride=1, padding=0)
        self.mean_linear = nn.Linear(z_dim * 12 * 12, z_dim)
        self.cat_op = nn.Linear(2*z_dim, z_dim)

        if use_bn:
            self.mean_conv_bn = nn.BatchNorm2d(z_dim)
            self.mean_linear_bn = nn.BatchNorm1d(z_dim)
            self.cat_bn = nn.BatchNorm1d(z_dim)

    def forward(self, x, eps=None):
        mean = self.mean_conv(x)
        if self.use_bn:
            mean = self.mean_conv_bn(mean)
        mean = self.activation(mean)

        mean = self.mean_linear(mean.view(mean.size(0), -1))
        if self.use_bn:
            mean = self.mean_linear_bn(mean)
        mean = mean.unsqueeze(-1).unsqueeze(-1)


        if eps is None:
            eps = torch.randn(mean.shape).to(mean.device)
        else:
            assert mean.shape == eps.shape

        out = torch.cat((mean, eps), dim=1)
        out = self.cat_op(out.squeeze())
        if self. use_bn:
            out = self.cat_bn(out)
        out = self.activation(out)

        return out

class Stochastic(nn.Module):
    def __init__(self, feature_dim, z_dim, is_log_var=True, use_bn=False, eps=1e-8, activation=gelu):
        super(Stochastic, self).__init__()
        self.is_log_var = is_log_var
        self.small_number = eps
        self.use_bn = use_bn
        self.activation = activation

        self.mean_conv = nn.Conv2d(feature_dim, z_dim, kernel_size=5, stride=1, padding=0)
        self.mean_linear = nn.Linear(z_dim * 12 * 12, z_dim)

        self.var_conv = nn.Conv2d(feature_dim, z_dim, kernel_size=5, stride=1, padding=0)
        self.var_linear = nn.Linear(z_dim * 12 * 12, z_dim)

        if use_bn:
            self.mean_conv_bn = nn.BatchNorm2d(z_dim)
            self.mean_linear_bn = nn.BatchNorm1d(z_dim)
            self.var_conv_bn = nn.BatchNorm2d(z_dim)
            self.var_linear_bn = nn.BatchNorm1d(z_dim)

    def forward(self, x, eps=None):
        mean = self.mean_conv(x)
        var = self.var_conv(x)
        if self.use_bn:
            mean = self.mean_conv_bn(mean)
            var = self.var_conv_bn(var)
        mean = self.activation(mean)
        var = self.activation(var)

        mean = self.mean_linear(mean.view(mean.size(0), -1))
        var = self.var_linear(var.view(var.size(0), -1))
        if self.use_bn:
            mean = self.mean_linear_bn(mean)
            var = self.var_linear_bn(var)
        mean = mean.unsqueeze(-1).unsqueeze(-1)
        var = var.unsqueeze(-1).unsqueeze(-1)

        if eps is None:
            eps = torch.randn(mean.shape).to(mean.device)
        else:
            assert mean.shape == eps.shape

        var = torch.nn.functional.softplus(var) + self.small_number
        var = var.clamp(0., 10.)

        z = stochastic_gaussian(mean, var, is_log_var=self.is_log_var, eps=eps)
        return z


class Deterministic(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, activation=gelu):
        super(Deterministic, self).__init__()

        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1)
        if self.use_bn:
            self.bn = nn.BatchNorm2d(out_dim)
            self.bn2 = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        out = self.conv(x)
        if self.use_bn:
            out = self.bn(out)
        out = self.activation(out)

        out = self.conv2(out)
        if self.use_bn:
            out = self.bn2(out)
        out = self.activation(out)

        out = out + x
        return out


class Projection(nn.Module):
    def __init__(self, z_dim, ngf=16, coef=4, use_bn=False, activation=gelu):
        super(Projection, self).__init__()

        self.use_bn = use_bn
        self.activation = activation
        self.ngf = 16
        self.coef = 4

        self.linear = nn.Linear(z_dim, coef * ngf * ngf)
        self.deconv1 = nn.ConvTranspose2d(coef, ngf * coef, kernel_size=5, stride=1, padding=2, bias=False)

        if self.use_bn:
            self.linear_bn = nn.BatchNorm1d(coef * ngf * ngf)
            self.deconv1_bn = nn.BatchNorm2d(ngf * coef)

    def forward(self, z):
        out = self.linear(z.view(z.size(0), -1))
        if self.use_bn:
            out = self.linear_bn(out)
        out = self.activation(out)

        out = self.deconv1(out.view(z.size(0), self.coef, self.ngf, self.ngf).contiguous())
        if self.use_bn:
            out = self.deconv1_bn(out)
        out = self.activation(out)
        return out


class G(nn.Module):
    def __init__(self, args, nc=3, ngf=16, coef=4):
        super(G, self).__init__()
        self.args = args
        self.ngf = ngf

        self.projection_layers = nn.ModuleList([Projection(z_dim, ngf=ngf, coef=coef, use_bn=self.args.use_bn) for z_dim in args.z_dims])
        self.deterministic_layers = nn.ModuleList([Deterministic(ngf * coef, ngf * coef, use_bn=self.args.use_bn) for _ in args.z_dims])
        self.deterministic_layers_extra = Deterministic(ngf * coef, ngf * coef, use_bn=self.args.use_bn)

        if self.args.use_ladder:
            self.stochastic_layers = nn.ModuleList([LadderStochastic(ngf * coef, z_dim, use_bn=self.args.use_bn) for z_dim in args.z_dims[1:]])
        else:
            self.stochastic_layers = nn.ModuleList([Stochastic(ngf * coef, z_dim, use_bn=self.args.use_bn) for z_dim in args.z_dims[1:]])

        self.output = nn.ConvTranspose2d(ngf * coef, nc, kernel_size=4, stride=2, padding=1)  # (64, 16, 16) -> (64, 32, 32)

    def forward(self, z_top, z_lowers):
        out = z_top
        for i, _ in enumerate(self.args.z_dims):
            out = self.projection_layers[i](out)
            if self.args.use_skip:
                if i > 0:
                    out = self.deterministic_layers[i](out) + out_det
                else:
                    out = self.deterministic_layers[i](out)
                out_det = out
            else:
                out = self.deterministic_layers[i](out)
            if i == len(self.args.z_dims) - 1:
                break
            out = self.stochastic_layers[i](out, z_lowers[i])

        out = self.deterministic_layers_extra(out)
        out = self.output(out)
        out = F.tanh(out)

        return out


def infer_z_single(g, z, x, args):
    mse = nn.MSELoss(reduction='sum')
    for i in range(args.z_n_iters):
        z_s_var = [torch.autograd.Variable(_z, requires_grad=True) for _z in z]
        z_top = z_s_var[0]
        z_lowers = z_s_var[1:]
        x_hat = g(z_top, z_lowers)
        L = 1.0 / (2.0 * args.z_sigma * args.z_sigma) * mse(x_hat, x)
        z_grads = torch.autograd.grad(L, z_s_var, retain_graph=True)

        for j, (z_var, z_grad) in enumerate(zip(z_s_var, z_grads)):
            z_var.data -= 0.5 * args.z_step_size * args.z_step_size * (z_var + z_grad)
            if args.z_with_noise:
                eps = torch.randn(*z_var.shape).to(z_var.device)
                z_var.data += args.z_step_size * eps

    z_s_k = [z.detach() for z in z_s_var]

    return z_s_k, z_grads


def infer_z(g, z, x, args):
    mse = nn.MSELoss(reduction='sum')
    for i in range(args.z_n_iters):
        z_s_var = [torch.autograd.Variable(_z, requires_grad=True) for _z in z]
        z_top = z_s_var[0]
        z_lowers = z_s_var[1:]
        x_hat = g(z_top, z_lowers)
        L = mse(x_hat, x)
        z_grads = torch.autograd.grad(L, z_s_var, retain_graph=True)

        for k, (z_var, z_grad) in enumerate(zip(z_s_var, z_grads)):
            z_var.data -= 0.5 * args.z_step_size[k] * args.z_step_size[k] * ((1.0 / (2.0 * args.z_sigma[k] * args.z_sigma[k]) * z_grad) + z_var)
            if args.z_with_noise:
                eps = torch.randn(*z_var.shape).to(z_var.device)
                z_var.data += args.z_step_size[k] * eps

    return z_s_var, z_grads

########################################################################################################

def grad_norm(model):
    return torch.sqrt(torch.sum(torch.tensor([torch.sum(p.grad ** 2) for p in model.parameters() if p.grad is not None])))

########################################################################################################

from fid_v2_tf_cpu import fid_score

def is_xsede():
    import socket
    return 'psc' in socket.gethostname()

def compute_fid(args, x_data, x_samples, use_cpu=False):

    assert type(x_data) == np.ndarray
    assert type(x_samples) == np.ndarray

    # RGB
    assert x_data.shape[3] == 3
    assert x_samples.shape[3] == 3

    # NHWC
    assert x_data.shape[1] == x_data.shape[2]
    assert x_samples.shape[1] == x_samples.shape[2]

    # [0,255]
    assert np.min(x_data) > 0.-1e-4
    assert np.max(x_data) < 255.+1e-4
    assert np.mean(x_data) > 10.

    # [0,255]
    assert np.min(x_samples) > 0.-1e-4
    assert np.max(x_samples) < 255.+1e-4
    assert np.mean(x_samples) > 1.

    if use_cpu:
        def create_session():
            import tensorflow.compat.v1 as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.0
            config.gpu_options.visible_device_list = ''
            return tf.Session(config=config)
    else:
        def create_session():
            import tensorflow.compat.v1 as tf
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.per_process_gpu_memory_fraction = 0.2
            config.gpu_options.visible_device_list = str(args.device)
            return tf.Session(config=config)

    path = '/tmp' if not is_xsede() else '/pylon5/ac561ep/enijkamp/inception'

    fid = fid_score(create_session, x_data, x_samples, path, cpu_only=use_cpu)

    return fid

def compute_fid_nchw(args, x_data, x_samples):

    to_nhwc = lambda x: np.transpose(x, (0, 2, 3, 1))

    x_data_nhwc = to_nhwc(255 * x_data)
    x_samples_nhwc = to_nhwc(255 * x_samples)

    fid = compute_fid(args, x_data_nhwc, x_samples_nhwc)

    return fid

########################################################################################################

def onehot(k):
    """
    Converts a number to its one-hot or 1-of-k representation
    vector.
    :param k: (int) length of vector
    :return: onehot function
    """
    def encode(label):
        y = torch.zeros(k)
        if label < k:
            y[label] = 1
        return y
    return encode


def get_mnist(location="./data/mnist", batch_size=64, labels_per_class=100, n_labels=10):
    from functools import reduce
    from operator import __or__
    from torch.utils.data.sampler import SubsetRandomSampler
    from torchvision.datasets import MNIST
    import torchvision.transforms as transforms

    # flatten_bernoulli = lambda x: transforms.ToTensor()(x)
    mnist_train = MNIST(location, train=True, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]), target_transform=onehot(n_labels))
    mnist_valid = MNIST(location, train=False, download=True, transform=transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()]), target_transform=onehot(n_labels))

    def get_sampler(labels, n=None):
        # Only choose digits in n_labels
        (indices,) = np.where(reduce(__or__, [labels == i for i in np.arange(n_labels)]))

        # Ensure uniform distribution of labels
        np.random.shuffle(indices)
        indices = np.hstack([list(filter(lambda idx: labels[idx] == i, indices))[:n] for i in range(n_labels)])

        indices = torch.from_numpy(indices)
        sampler = SubsetRandomSampler(indices)
        return sampler

    # Dataloaders for MNIST
    labelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=False, sampler=get_sampler(mnist_train.train_labels.numpy(), labels_per_class))
    unlabelled = torch.utils.data.DataLoader(mnist_train, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=False, sampler=get_sampler(mnist_train.train_labels.numpy()))
    validation = torch.utils.data.DataLoader(mnist_valid, batch_size=batch_size, drop_last=True, num_workers=0, pin_memory=False, sampler=get_sampler(mnist_valid.test_labels.numpy()))

    return labelled, unlabelled, validation


def get_celeb(img_size=32, batch_size=64):
    data_path = './data/celeba/img_align_celeba'
    cache_pkl = './data/celeba/celeba_40000_32.pickle'

    from data import SingleImagesFolderMTDataset
    import PIL
    import torchvision.transforms as transforms

    dataset = SingleImagesFolderMTDataset(root=data_path,
                                          cache=cache_pkl,
                                          num_images=40000,
                                          transform=transforms.Compose([
                                              PIL.Image.fromarray,
                                              transforms.Resize(img_size),
                                              transforms.CenterCrop(img_size),
                                              transforms.ToTensor(),
                                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                          ]))

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=int(0))

    return dataloader

########################################################################################################

def set_lr(optim, lr):
    for param_group in optim.param_groups:
        param_group['lr'] = lr


def train(args_job, output_dir_job, output_dir, return_dict):

    # preamble
    args = parse_args()
    args = pygrid.overwrite_opt(args, args_job)
    # TODO(nijkamp) cleanup
    args.z_step_size = [args.z_step_size] * 5
    z_sigma_init = args.z_sigma

    args = to_named_dict(args)

    set_gpu(args.device)
    set_seed(args.seed)

    job_id = int(args['job_id'])

    logger = pygrid.setup_logging('job{}'.format(job_id), output_dir, console=True)
    logger.info(args)

    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')

    ds_train = get_celeb(batch_size=args.n_batch)

    to_range_0_1 = lambda x: (x + 1.) / 2.
    ds_fid = np.array(torch.cat([to_range_0_1(x) for x in iter(ds_train)]).cpu().numpy())

    def sample_p_0():
        return [torch.randn(*[x.shape[0], z_shape, 1, 1], device=device) for i, z_shape in enumerate(args.z_dims)]

    sqrt = lambda x: int(torch.sqrt(torch.Tensor([x])))
    plot = lambda p, x: torchvision.utils.save_image(torch.clamp(x, 0., 1.), p, normalize=True, nrow=sqrt(args.n_batch))

    g = G(args).to(device)

    optim = torch.optim.Adam(g.parameters(), lr=args.lr, betas=[args.beta1, .999])

    mse = nn.MSELoss(reduction='sum')

    return_dict['stats'] = {'epoch': 0, 'i': 0, 'mse': 0.0, 'mse_best': 0.0, 'fid': 0.0, 'fid_best': 0.0, 'success': 0}

    fid = 0.0
    fid_best = math.inf
    mse_best = math.inf

    l = 0

    lr = args.lr

    i_iter = 0

    for epoch in range(args.n_epochs+1):

        if args.lr_decay == 0:
            pass

        if args.lr_decay == 1:
            if epoch > 0 and epoch == 50:
                lr = args.lr / 2
                set_lr(optim, lr)

        if args.lr_decay == 2:
            lr = args.lr * max(0.05, 1. - (epoch / args.n_epochs))
            set_lr(optim, lr)

        if args.lr_decay == 3:
            if epoch > 40:
                lr = 0.6 * args.lr
            if epoch > 60:
                lr = 0.4 * args.lr
            if epoch > 80:
                lr = 0.2 * args.lr
            set_lr(optim, lr)

        if args.lr_decay == 4:
            if epoch > 0 and epoch == 50:
                lr = args.lr / 2
                set_lr(optim, lr)
            if epoch > 0 and epoch == 200:
                lr = args.lr / 4
                set_lr(optim, lr)

        if args.lr_decay == 5:
            if epoch > 0 and epoch == 50:
                lr = args.lr / 2
                set_lr(optim, lr)
            if epoch > 0 and epoch == 100:
                lr = args.lr / 4
                set_lr(optim, lr)

        if args.lr_decay == 6:
            if epoch > 0 and epoch == 40:
                lr = args.lr / 2
                set_lr(optim, lr)
            if epoch > 0 and epoch == 60:
                lr = args.lr / 3
                set_lr(optim, lr)
            if epoch > 0 and epoch == 80:
                lr = args.lr / 4
                set_lr(optim, lr)

        if args.lr_decay == 7:
            if epoch > 0 and epoch == 50:
                lr = args.lr / 2
                set_lr(optim, lr)
            if epoch > 0 and epoch == 100:
                lr = args.lr / 4
                set_lr(optim, lr)
            if epoch > 0 and epoch == 300:
                lr = args.lr / 8
                set_lr(optim, lr)


        for i, x in enumerate(ds_train):

            i_iter += 1

            if args.z_sigma_decay == -1:

                factor = 0.0

            if args.z_sigma_decay == 0:

                factor = 0.0
                if epoch > 1:
                    factor = 0.005
                if epoch > 2:
                    factor = 0.01
                if epoch > 4:
                    factor = 0.02
                if epoch > 8:
                    factor = 0.03
                if epoch > 12:
                    factor = 0.04
                if epoch > 16:
                    factor = 0.05
                if epoch > 20:
                    factor = 0.06
                if epoch > 24:
                    factor = 0.07
                if epoch > 30:
                    factor = 0.08
                if epoch > 36:
                    factor = 0.09
                if epoch > 40:
                    factor = 0.10

            if args.z_sigma_decay == 1:

                factor = 0.0
                if epoch > 1:
                    factor = 0.005
                if epoch > 3:
                    factor = 0.01
                if epoch > 7:
                    factor = 0.02
                if epoch > 10:
                    factor = 0.03
                if epoch > 15:
                    factor = 0.04
                if epoch > 20:
                    factor = 0.05
                if epoch > 25:
                    factor = 0.06
                if epoch > 30:
                    factor = 0.07
                if epoch > 40:
                    factor = 0.08
                if epoch > 50:
                    factor = 0.09
                if epoch > 60:
                    factor = 0.10

            if args.z_sigma_decay == 2:

                factor = 0.0
                if epoch > 5:
                    factor = 0.005
                if epoch > 10:
                    factor = 0.01
                if epoch > 15:
                    factor = 0.02
                if epoch > 20:
                    factor = 0.03
                if epoch > 25:
                    factor = 0.04
                if epoch > 30:
                    factor = 0.05
                if epoch > 35:
                    factor = 0.06
                if epoch > 40:
                    factor = 0.07
                if epoch > 50:
                    factor = 0.08
                if epoch > 60:
                    factor = 0.09
                if epoch > 70:
                    factor = 0.10

            if args.z_sigma_decay == 3:

                if epoch > 5:
                    assert (z_sigma_init - 0.15) > 0.
                    factor = (z_sigma_init - 0.15) * ((epoch-5) / (args.n_epochs-5))

            if args.z_sigma_decay == 4:

                factor = 0.0
                if epoch > 5:
                    factor = 0.005 / 2.
                if epoch > 10:
                    factor = 0.01 / 2.
                if epoch > 15:
                    factor = 0.02 / 2.
                if epoch > 20:
                    factor = 0.03 / 2.
                if epoch > 25:
                    factor = 0.04 / 2.
                if epoch > 30:
                    factor = 0.05 / 2.
                if epoch > 35:
                    factor = 0.06 / 2.
                if epoch > 40:
                    factor = 0.07 / 2.
                if epoch > 50:
                    factor = 0.08 / 2.
                if epoch > 60:
                    factor = 0.09 / 2.
                if epoch > 70:
                    factor = 0.10 / 2.

            if args.z_sigma_decay == 5:

                factor = 0.0
                if epoch > 5:
                    factor = 0.005
                if epoch > 10:
                    factor = 0.01
                if epoch > 15:
                    factor = 0.02
                if epoch > 20:
                    factor = 0.03
                if epoch > 25:
                    factor = 0.04
                if epoch > 30:
                    factor = 0.05
                if epoch > 35:
                    factor = 0.06
                if epoch > 40:
                    factor = 0.07
                if epoch > 45:
                    factor = 0.08
                if epoch > 50:
                    factor = 0.09
                if epoch > 55:
                    factor = 0.10
                if epoch > 60:
                    factor = 0.11
                if epoch > 65:
                    factor = 0.12
                if epoch > 70:
                    factor = 0.13
                if epoch > 75:
                    factor = 0.14
                if epoch > 80:
                    factor = 0.15
                if epoch > 85:
                    factor = 0.16
                if epoch > 90:
                    factor = 0.17
                if epoch > 95:
                    factor = 0.18
                if epoch > 100:
                    factor = 0.19
                if epoch > 105:
                    factor = 0.20

            if args.z_sigma_decay == 6:

                factor = 0.0
                if epoch > 5:
                    factor = 0.005
                if epoch > 10:
                    factor = 0.01
                if epoch > 15:
                    factor = 0.02
                if epoch > 20:
                    factor = 0.03
                if epoch > 25:
                    factor = 0.04
                if epoch > 30:
                    factor = 0.05
                if epoch > 35:
                    factor = 0.06
                if epoch > 40:
                    factor = 0.07
                if epoch > 45:
                    factor = 0.08
                if epoch > 50:
                    factor = 0.09
                if epoch > 55:
                    factor = 0.10
                if epoch > 60:
                    factor = 0.11
                if epoch > 65:
                    factor = 0.12
                if epoch > 70:
                    factor = 0.13
                if epoch > 75:
                    factor = 0.14
                if epoch > 80:
                    factor = 0.15
                if epoch > 85:
                    factor = 0.16
                if epoch > 90:
                    factor = 0.17
                if epoch > 95:
                    factor = 0.18
                if epoch > 100:
                    factor = 0.19
                if epoch > 105:
                    factor = 0.20
                if epoch > 110:
                    factor = 0.21
                if epoch > 115:
                    factor = 0.22
                if epoch > 120:
                    factor = 0.23
                if epoch > 125:
                        factor = 0.24
                if epoch > 130:
                        factor = 0.25

            if args.z_sigma_decay == 7:

                factor = 0.0
                if epoch > 5:
                    factor = 0.005
                if epoch > 10:
                    factor = 0.01
                if epoch > 15:
                    factor = 0.02
                if epoch > 20:
                    factor = 0.03
                if epoch > 25:
                    factor = 0.04
                if epoch > 30:
                    factor = 0.05
                if epoch > 35:
                    factor = 0.06
                if epoch > 40:
                    factor = 0.07
                if epoch > 45:
                    factor = 0.08
                if epoch > 50:
                    factor = 0.09
                if epoch > 55:
                    factor = 0.10
                if epoch > 60:
                    factor = 0.11
                if epoch > 65:
                    factor = 0.12
                if epoch > 70:
                    factor = 0.13
                if epoch > 75:
                    factor = 0.14
                if epoch > 80:
                    factor = 0.15

            if args.z_sigma_decay == 8:

                factor = 0.0
                if epoch > 5:
                    factor = 0.005
                if epoch > 10:
                    factor = 0.01
                if epoch > 15:
                    factor = 0.02
                if epoch > 20:
                    factor = 0.03
                if epoch > 25:
                    factor = 0.04
                if epoch > 30:
                    factor = 0.05
                if epoch > 35:
                    factor = 0.06
                if epoch > 40:
                    factor = 0.07
                if epoch > 45:
                    factor = 0.08
                if epoch > 50:
                    factor = 0.09
                if epoch > 55:
                    factor = 0.10
                if epoch > 60:
                    factor = 0.11
                if epoch > 65:
                    factor = 0.12
                if epoch > 70:
                    factor = 0.13
                if epoch > 75:
                    factor = 0.14
                if epoch > 80:
                    factor = 0.15
                if epoch > 85:
                    factor = 0.16
                if epoch > 90:
                    factor = 0.17
                if epoch > 95:
                    factor = 0.18
                if epoch > 100:
                    factor = 0.19
                if epoch > 105:
                    factor = 0.20

            if args.z_sigma_decay == 9:

                factor = 0.0
                if epoch > 5:
                    factor = 0.005
                if epoch > 10:
                    factor = 0.01
                if epoch > 12:
                    factor = 0.02
                if epoch > 14:
                    factor = 0.03
                if epoch > 16:
                    factor = 0.04
                if epoch > 18:
                    factor = 0.05
                if epoch > 20:
                    factor = 0.06
                if epoch > 22:
                    factor = 0.07
                if epoch > 24:
                    factor = 0.08
                if epoch > 26:
                    factor = 0.09
                if epoch > 28:
                    factor = 0.10
                if epoch > 30:
                    factor = 0.11
                if epoch > 32:
                    factor = 0.12
                if epoch > 34:
                    factor = 0.13
                if epoch > 36:
                    factor = 0.14
                if epoch > 38:
                    factor = 0.15
                if epoch > 40:
                    factor = 0.16
                if epoch > 42:
                    factor = 0.17
                if epoch > 44:
                    factor = 0.18
                if epoch > 46:
                    factor = 0.19
                if epoch > 48:
                    factor = 0.20
                if epoch > 50:
                    factor = 0.20

            args.z_sigma = [z_sigma_init - factor, z_sigma_init - factor, z_sigma_init - factor, z_sigma_init - factor, z_sigma_init - factor]

            x = x.to(device)
            z_0 = sample_p_0()
            z_init = [_z_0.clone() for _z_0 in z_0]
            z, z_grad = infer_z(g, z_init, x, args)
            z_top = z[0]
            z_lowers = z[1:]
            x_hat = g(z_top, z_lowers)
            mse_hat = mse(x, x_hat)
            # L = 0.5 * args.z_sigma * args.z_sigma * mse_hat
            L = mse_hat

            optim.zero_grad()
            L.backward()
            optim.step()


            if True and epoch > 0 and epoch % 5 == 0 and i == 0:
                torch.save(g.state_dict(), '%s/g_epoch_%d.pth' % (output_dir, epoch))
                torch.save(optim.state_dict(), '%s/optim_epoch_%d.pth' % (output_dir, epoch))

            if True and epoch > 0 and epoch % 10 == 0 and i == 0:
                try:
                    def sample_x():
                        z_samples = sample_p_0()
                        x_samples = to_range_0_1(g(z_samples[0], z_samples[1:])).clamp(min=0., max=1.).detach().cpu()
                        return x_samples
                    x_samples = torch.cat([sample_x() for _ in range(int(len(ds_fid) / x.shape[0]))]).numpy()
                    fid = compute_fid_nchw(args, ds_fid, x_samples)
                    if fid < fid_best:
                        fid_best = fid

                except Exception as e:
                    print(e)
                    logger.critical(e, exc_info=True)
                    logger.info('FID failed')

            if i % 10 == 0:
                z_s_0 = z_0
                z_s_k = z
                z_grads = z_grad

                z_norm_str = ' '.join(['[{:8.2f} {:8.2f}]'.format(torch.norm(z_0, dim=0).mean(), torch.norm(z_k, dim=0).mean()) for z_0, z_k in zip(z_s_0, z_s_k)])
                z_disp_str = ' '.join(['{:8.2f}'.format(torch.norm(z_0-z_k, dim=0).mean()) for z_0, z_k in zip(z_s_0, z_s_k)])
                z_grads_str = ' '.join(['{:8.2f}'.format(torch.norm(g, dim=0).mean()) for g in z_grads])

                w_max = np.max([p.abs().max() for p in g.parameters()])

                logger.info('{}  {:4d}/{:4d} {:4d}/{:4d}  lr={:8.5f}  mse={:12.2f}  mse_best={:12.2f}  fid={:12.6f}  fid_best={:12.6f}   sigma={}   w_grad={:12.4f} w_max={:12.4f}  z_norms={}  z_disp={}  z_grads={}'.format(job_id, epoch, args.n_epochs, i, len(ds_train), lr, mse_hat.cpu().data.numpy() / args.n_batch, mse_best, fid, fid_best, args.z_sigma, grad_norm(g), w_max, z_norm_str, z_disp_str, z_grads_str))

            if i == 0:
                plot('{}/{}_{:>06d}_x.png'.format(output_dir, epoch, i), to_range_0_1(x))
                plot('{}/{}_{:>06d}_x_hat.png'.format(output_dir, epoch, i), to_range_0_1(x_hat))
                z_samples = sample_p_0()
                plot('{}/{}_{:>06d}_x_sample.png'.format(output_dir, epoch, i), to_range_0_1(g(z_samples[0], z_samples[1:])))

            if torch.isnan(mse_hat):
                logger.info('early stop: nan')
                assert not torch.isnan(mse_hat)

            if False and epoch > 100 and (mse_hat / args.n_batch) > 200:
                logger.info('early stop: mse')
                assert not (mse_hat / args.n_batch) > 200

            mse_current = mse_hat.data.item() / args.n_batch
            if mse_current < mse_best:
                mse_best = mse_current

            return_dict['stats'] = {'epoch': epoch, 'i': i, 'mse': mse_current, 'mse_best': mse_best, 'fid': fid, 'fid_best': fid_best, 'success': 0}

    return_dict['stats'] = {'epoch': epoch, 'i': i, 'mse': mse_current, 'mse_best': mse_best, 'fid': fid, 'fid_best': fid_best, 'success': 1}

##############################################################################

def set_seed(seed=None):
    if seed is None:
        seed = random.randint(1, 10000)
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)


def set_gpu(gpu, deterministic=True):
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu)
        if not deterministic:
            torch.backends.cudnn.benchmark = True
        else:
            torch.backends.cudnn.deterministic = True

        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)


def get_exp_id(file):
    return os.path.splitext(os.path.basename(file))[0]


def get_output_dir(exp_id):
    t = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join('output/' + exp_id, t)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def setup_logging(name, output_dir, console=True):
    log_format = logging.Formatter("%(asctime)s : %(message)s")
    logger = logging.getLogger(name)
    logger.handlers = []
    output_file = os.path.join(output_dir, 'output.log')
    file_handler = logging.FileHandler(output_file)
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(log_format)
        logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)
    return logger


def copy_source(file, output_dir):
    shutil.copyfile(file, os.path.join(output_dir, os.path.basename(file)))

##############################################################################

import itertools


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def to_named_dict(ns):
    d = AttrDict()
    for (k, v) in zip(ns.__dict__.keys(), ns.__dict__.values()):
        d[k] = v
    return d


def merge_dicts(a, b, c):
    d = {}
    d.update(a)
    d.update(b)
    d.update(c)
    return d


def main():

    use_pygrid = True

    if use_pygrid:

        # TODO(nijkamp): enumerate gpu devices here
        device_ids = [0]
        workers = len(device_ids)

        # set devices
        pygrid.init_mp()
        pygrid.fill_queue(device_ids)

        fs_suffix = './' if not is_xsede() else '/home/enijkamp/pylon/short_run_abp/'

        # set opts
        get_opts_filename = lambda exp: fs_suffix + '{}.csv'.format(exp)
        exp_id = pygrid.get_exp_id(__file__)

        write_opts = lambda opts: pygrid.write_opts(opts, lambda: open(get_opts_filename(exp_id), mode='w'))
        read_opts = lambda: pygrid.read_opts(lambda: open(get_opts_filename(exp_id), mode='r'))

        output_dir = fs_suffix + pygrid.get_output_dir(exp_id)
        os.makedirs(output_dir + '/samples')

        if not os.path.exists(get_opts_filename(exp_id)):
            write_opts(create_args_grid())
        write_opts(pygrid.reset_job_status(read_opts()))

        # set logging
        logger = pygrid.setup_logging('main', output_dir, console=True)
        logger.info('available devices {}'.format(device_ids))

        # run
        copy_source(__file__, output_dir)
        pygrid.run_jobs(logger, exp_id, output_dir, workers, train, read_opts, write_opts, update_job_result)
        logger.info('done')

    else:

        # preamble
        exp_id = pygrid.get_exp_id(__file__)
        fs_suffix = './'
        output_dir = fs_suffix + pygrid.get_output_dir(exp_id)

        # run
        copy_source(__file__, output_dir)
        opt = {'job_id': int(0), 'status': 'open', 'device': 3}
        train(opt, output_dir, output_dir, {})


if __name__ == '__main__':
    main()
