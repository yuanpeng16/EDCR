import sys
sys.path.insert(0, '../..')

import torch
import numpy as np
import matplotlib.pyplot as plt

from causal_meta.modules.mdn import mdn_nll
from causal_meta.utils.data_utils import RandomSplineSCM
from causal_meta.utils.train_utils import train_nll, make_alpha
from models import mdn, gmm, auc_transfer_metric
from argparse import Namespace

import torch.nn.functional as F
from causal_meta.utils.torch_utils import logsumexp
from tqdm import tnrange

import time
from utils import write_data

torch.manual_seed(4)
np.random.seed(4)

def train_alpha(opt, model_x2y, model_y2x, model_g2y, model_g2x, alpha, gt_scm,
                distr, sweep_distr, nll, transfer_metric, mixmode='logmix'):
    # Everyone to CUDA
    if opt.CUDA:
        model_x2y.cuda()
        model_y2x.cuda()
    alpha_optim = torch.optim.Adam([alpha], lr=opt.ALPHA_LR)
    frames = []
    iterations = tnrange(opt.ALPHA_NUM_ITER, leave=False)
    start = time.time()
    for iter_num in iterations:
        # Sample parameter for the transfer distribution
        sweep_param = sweep_distr()
        # Sample X from transfer
        X_gt = distr(sweep_param)
        Y_gt = gt_scm(X_gt)
        with torch.no_grad():
            if opt.CUDA:
                X_gt, Y_gt = X_gt.cuda(), Y_gt.cuda()
        # Evaluate performance
        metric_x2y = transfer_metric(opt, model_x2y, model_g2x, X_gt, Y_gt, nll)
        metric_y2x = transfer_metric(opt, model_y2x, model_g2y, Y_gt, X_gt, nll)
        # Estimate gradient
        if mixmode == 'logmix':
            loss_alpha = torch.sigmoid(alpha) * metric_x2y + (1 - torch.sigmoid(alpha)) * metric_y2x
        else:
            log_alpha, log_1_m_alpha = F.logsigmoid(alpha), F.logsigmoid(-alpha)
            as_lse = logsumexp(log_alpha + metric_x2y, log_1_m_alpha + metric_y2x)
            if mixmode == 'logsigp':
                loss_alpha = as_lse
            elif mixmode == 'sigp':
                loss_alpha = as_lse.exp()
        # Optimize
        alpha_optim.zero_grad()
        loss_alpha.backward()
        alpha_optim.step()
        end = time.time()

        with torch.no_grad():
            gamma = torch.sigmoid(alpha).item()

        if alpha.item() > 0:
            prediction = 1.
        else:
            prediction = 0.

        # Append info
        with torch.no_grad():
            frames.append(Namespace(iter_num=iter_num,
                                    alpha=gamma,
                                    sig_alpha=prediction,
                                    time=end-start,
                                    metric_x2y=metric_x2y,
                                    metric_y2x=metric_y2x,
                                    loss_alpha=loss_alpha.item()))
        iterations.set_postfix(alpha='{0:.4f}'.format(torch.sigmoid(alpha).item()))
    return frames

def normal(mean, std, N):
    return torch.normal(torch.ones(N).mul_(mean), torch.ones(N).mul_(std)).view(-1, 1)

def filtered_append(entry, entry_list):
    flag = True
    for a in entry:
        if np.isnan(a):
            flag = False
            break
    if flag:
        entry_list.append(entry)
    return entry_list

opt = Namespace()
# Model
opt.CAPACITY = 32
opt.NUM_COMPONENTS = 10
opt.GMM_NUM_COMPONENTS = 10
# Training
opt.LR = 0.001
opt.NUM_ITER = 3000
opt.CUDA = False
opt.REC_FREQ = 10
# Meta
opt.ALPHA_LR = 0.1
opt.ALPHA_NUM_ITER = 50
opt.FINETUNE_LR = 0.001
opt.FINETUNE_NUM_ITER = 10
opt.PARAM_DISTRY = lambda mean: normal(mean, 2, opt.NUM_TRANS_SAMPLES)
opt.PARAM_SAMPLER = lambda: np.random.uniform(-4, 4)
# Sampling
opt.NUM_SAMPLES = 1000
opt.NUM_TRANS_SAMPLES = 1000
opt.TRAIN_DISTRY = lambda: normal(0, 2, opt.NUM_SAMPLES)
#opt.TRANS_DISTRY = lambda: normal(random.randint(-4, 4), 2, opt.NUM_SAMPLES)

alpha_list = []
beta_list = []
gamma_list = []
iterations = 100
for i in range(iterations):
    scm = RandomSplineSCM(False, True, 8, 10, 3, range_scale=1.)

    model_x2y = mdn(opt)
    frames_x2y = train_nll(opt, model_x2y, scm, opt.TRAIN_DISTRY, polarity='X2Y',
        loss_fn=mdn_nll, decoder=None, encoder=None)

    model_y2x = mdn(opt)
    frames_y2x = train_nll(opt, model_y2x, scm, opt.TRAIN_DISTRY, polarity='Y2X',
        loss_fn=mdn_nll, decoder=None, encoder=None)

    alpha = make_alpha(opt)
    alpha_frames = train_alpha(opt, model_x2y, model_y2x, None, None, alpha, scm,
                               opt.PARAM_DISTRY, opt.PARAM_SAMPLER, mdn_nll,
                               auc_transfer_metric, mixmode='logsigp')

    alphas = np.asarray([frame.sig_alpha for frame in alpha_frames])
    betas = np.asarray([frame.time for frame in alpha_frames])
    gammas = np.asarray([frame.alpha for frame in alpha_frames])

    alpha_list.append(alphas)
    beta_list.append(betas)
    gamma_list = filtered_append(gammas, gamma_list)
    print(i, '/', iterations)

mean_alpha = np.mean(alpha_list, axis=0)
mean_beta = np.mean(beta_list, axis=0)
mean_gamma = np.mean(gamma_list, axis=0)

write_data("results", "baseline.txt", mean_gamma, mean_alpha, mean_beta)

for i, (a, b, c) in enumerate(zip(mean_gamma, mean_alpha, mean_beta)):
    print(i + 1, a, b, c)

fig = plt.figure(figsize=(9, 5))
ax = plt.subplot(1, 1, 1)

ax.tick_params(axis='both', which='major', labelsize=13)
ax.axhline(1, c='lightgray', ls='--')
ax.axhline(0, c='lightgray', ls='--')
ax.plot(alphas, lw=2, color='k', label='N = {0}'.format(10))

ax.set_xlim([0, opt.ALPHA_NUM_ITER - 1])
ax.set_xlabel('Number of episodes', fontsize=14)
ax.set_ylabel(r'$\sigma(\gamma)$', fontsize=14)
ax.legend(loc=4, prop={'size': 13})

plt.show()