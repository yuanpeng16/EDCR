import sys
sys.path.append('../..')

import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy import interpolate

import random
import tqdm
from copy import deepcopy
from argparse import Namespace
import warnings

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from causal_meta.utils.data_utils import RandomSplineSCM

from causal_meta.utils import train_utils as tu
from encoder import Rotor
from causal_meta.modules.mdn import MDN, GMM, mdn_nll
from causal_meta.modules.gmm import GaussianMixture

SEED = 91023
torch.manual_seed(SEED)
np.random.seed(SEED)

def normal(mean, std, N):
    return torch.normal(torch.ones(N).mul_(mean), torch.ones(N).mul_(std)).view(-1, 1)

def normal_like(X):
    mean = X.mean()
    std = X.std()
    return normal(mean, std, X.size(0))

def mlp(opt):
    if opt.NUM_MID_LAYERS == -1:
        return nn.Linear(opt.INP_DIM, opt.OUT_DIM)
    else:
        return nets.MLP(opt.INP_DIM, opt.OUT_DIM, opt.NUM_MID_LAYERS,
                        opt.CAPACITY, opt.INP_NOISE)

def mdn(opt):
    return MDN(opt.CAPACITY, opt.NUM_COMPONENTS)

def gmm(opt):
    return GaussianMixture(opt.GMM_NUM_COMPONENTS)

def xcodergen(opt):
    # Make rotor
    return Rotor(opt.XCODER_INIT)

# Test
rand_scm = RandomSplineSCM(False, True, 8, 8, 3, range_scale=1.)

def plot_key(frames, key, show=True, label=None, name=None):
    its, vals = zip(*[(frame.iter_num, getattr(frame, key)) for frame in frames])
    if show:
        plt.figure()
    plt.plot(its, vals, label=label)
    if show:
        plt.xlabel("Iterations")
        plt.ylabel(name if name is not None else key.title())
        plt.show()

def gradnan_filter(model):
    nan_found = False
    for p in model.parameters():
        if p.grad is None:
            continue
        nan_mask = torch.isnan(p.grad.data)
        nan_found = bool(nan_mask.any().item())
        p.grad.data[nan_mask] = 0.
    return nan_found

def marginal_nll(opt, inp, nll):
    model_g = gmm(opt)
    if opt.CUDA:
        model_g = model_g.cuda()
    model_g.fit(inp, n_iter=opt.EM_ITERS)
    with torch.no_grad():
        loss_marginal = nll(model_g(inp), inp)
    return loss_marginal

def process_sample(X, model_x2y, model_y2x, scm, encoder, decoder):
    Y = scm(X)
    if opt.CUDA:
        X, Y = X.cuda(), Y.cuda()
    # Decode
    with torch.no_grad():
        X, Y = decoder(X, Y)
    # Encode
    X, Y = encoder(X, Y)

    # Evaluate total regret
    loss_x2y = mdn_nll(model_x2y(X), Y)
    loss_y2x = mdn_nll(model_y2x(Y), X)

    if torch.isnan(loss_x2y).item() or torch.isnan(loss_y2x).item():
        raise()

    return loss_x2y, loss_y2x

def get_transfer_loss(opt, model_x2y, model_y2x, scm, encoder, decoder, alpha):
    loss_x2y, loss_y2x = process_sample(opt.TRANS_DISTRY(), model_x2y, model_y2x, scm, encoder, decoder)
    return loss_x2y, loss_y2x

def get_train_loss(opt, model_x2y, model_y2x, scm, encoder, decoder):
    loss_x2y, loss_y2x = process_sample(opt.TRAIN_DISTRY(), model_x2y, model_y2x, scm, encoder, decoder)
    return loss_x2y, loss_y2x

def encoder_train_shared_regret(opt, model_x2y, model_y2x, scm, encoder, decoder, alpha):
    if opt.CUDA:
        model_x2y = model_x2y.cuda()
        model_y2x = model_y2x.cuda()
        encoder = encoder.cuda()
        decoder = decoder.cuda()

    params = list(encoder.parameters())
    if opt.USE_GAMMA:
        params.append(alpha)
    transfer_optim= torch.optim.Adam(params, opt.TRANSFER_LR)

    frames = []
    start = time.time()
    for meta_iter in range(opt.NUM_META_ITER):
        # Preheat the models
        train_info_x2y = tu.train_nll(opt, model_x2y, scm, opt.TRAIN_DISTRY, 'X2Y',
                         mdn_nll, decoder, encoder)
        train_info_y2x = tu.train_nll(opt, model_y2x, scm, opt.TRAIN_DISTRY, 'Y2X',
                         mdn_nll, decoder, encoder)

        train_loss_x2y = train_info_x2y[-1].loss
        train_loss_y2x = train_info_y2x[-1].loss

        transfer_loss_x2y, transfer_loss_y2x = get_transfer_loss(opt, model_x2y, model_y2x, scm, encoder, decoder, alpha)
        transfer_loss_x2y -= train_loss_x2y
        transfer_loss_y2x -= train_loss_y2x
        if opt.USE_GAMMA:
            sig = F.logsigmoid(alpha)
            transfer_loss = torch.log(sig * torch.exp(transfer_loss_x2y) + (1 - sig) * torch.exp(transfer_loss_y2x))
        else:
            transfer_loss = torch.min(transfer_loss_x2y, transfer_loss_y2x)

        transfer_optim.zero_grad()
        transfer_loss.backward()
        transfer_optim.step()

        end = time.time()

        frames.append(Namespace(iter_num=meta_iter,
                                regret_x2y=train_loss_x2y,
                                regret_y2x=train_loss_y2x,
                                loss=transfer_loss.item(),
                                alpha=alpha.item(),
                                theta=encoder.theta.item(),
                                como_time=end-start))
        if meta_iter % 1 == 0:
            loss = transfer_loss
            print(meta_iter, loss.item(), encoder.theta.item(), alpha.item(), transfer_loss.item())

    return frames


def plot_theta(frames, gt_theta, save=False):
    its, vals = zip(*[(frame.iter_num, frame.theta / (np.pi / 2)) for frame in frames])
    tits, tvals = zip(*[(frame.iter_num, frame.como_time) for frame in frames])
    for i, v, t in zip(its, vals, tvals):
        print(i + 1, v, t)
    gt_theta = -gt_theta.item() / (np.pi / 2)
    plt.figure()
    plt.plot(its, vals, label=r'$\theta_{\mathcal{E}}$')
    plt.plot(its, [gt_theta] * len(its), linestyle='--', label=r'Solution 1 $\left(+\frac{\pi}{4}\right)$')
    plt.plot(its, [gt_theta - 1] * len(its), linestyle='--', label=r'Solution 2 $\left(-\frac{\pi}{4}\right)$')
    plt.xlabel("Iterations")
    plt.ylabel("Encoder Angle [π/2 rad]")
    plt.legend()
    if save:
        plt.savefig('fixed-encoder-evo.pdf', bbox_inches='tight', format='pdf')
    plt.show()


def probe_xcoders(encoder, decoder):
    # Test encoder and decoder
    with torch.no_grad():
        _X = torch.tensor([[1.]])
        _Y = torch.tensor([[0.]])
        if opt.CUDA:
            _X, _Y = _X.to(encoder.theta.device), _Y.to(encoder.theta.device)
        _X_d, _Y_d = decoder(_X, _Y)
        _X_de, _Y_de = encoder(_X_d, _Y_d)
    print(f"Initial (A, B) = {_X.item()}, {_Y.item()}")
    print(f"Decoded (X, Y) = {_X_d.item()}, {_Y_d.item()}")
    print(f"Encoded (U, V) = {_X_de.item()}, {_Y_de.item()}")

opt = Namespace()

# Model
opt.CAPACITY = 32
opt.NUM_COMPONENTS = 10
opt.GMM_NUM_COMPONENTS = 10

# Training
opt.LR = 0.01
opt.NUM_ITER = 20
opt.NUM_META_ITER = 1000
opt.TRAIN_LR = 0.01
opt.TRANSFER_LR = 0.01
opt.CUDA = True
opt.REC_FREQ = 10
opt.ALPHA_INIT = 0.
opt.USE_BASELINE = False

# Fine tuning
opt.FINETUNE_NUM_ITER = 5
opt.EM_ITERS = 500

# Sampling
opt.NUM_SAMPLES = 1000
opt.TRAIN_DISTRY = lambda: normal(0, 2, opt.NUM_SAMPLES)
opt.TRANS_DISTRY = lambda: normal(np.random.uniform(-4, 4),
                                  2, opt.NUM_SAMPLES)

opt.USE_GAMMA = False

# Encoder
opt.DECODER_DEFAULT = -float(0.5 * np.pi/2)

gt_decoder = Rotor(opt.DECODER_DEFAULT)

encoder = Rotor(0. * np.pi/2)

probe_xcoders(encoder, gt_decoder)

alpha = tu.make_alpha(opt)

model_x2y = mdn(opt)
model_y2x = mdn(opt)

frames = encoder_train_shared_regret(opt, model_x2y, model_y2x, rand_scm, encoder, gt_decoder, alpha)

plot_theta(frames, gt_decoder.theta)

if opt.USE_GAMMA:
    plot_key(frames, 'alpha', name='Structural Parameter')
