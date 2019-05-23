import sys
sys.path.insert(0, '../..')

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm_notebook

from causal_meta.utils.data_utils import generate_data_categorical
from proposed_models import StructuralModel

# fix data
np.random.seed(4)

N = 10
model = StructuralModel(N, dtype=torch.float64)

num_episodes = 100
batch_size = 100 # 1
num_test = 10000
num_training = 10 # 100
num_transfers = 10 # 100

def get_gradient_norm(model, loss):
    model.zero_grad()
    loss.backward()
    total_norm = 0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1. / 2)
    return total_norm

losses = np.zeros((2, num_training, num_transfers, num_episodes))

for k in tnrange(num_training):
    pi_A_1 = np.random.dirichlet(np.ones(N))
    pi_B_A = np.random.dirichlet(np.ones(N), size=N)
    for j in tnrange(num_transfers, leave=False):
        model.set_ground_truth(pi_A_1, pi_B_A)
        pi_A_2 = np.random.dirichlet(np.ones(N))
        all_x_transfer = torch.from_numpy(generate_data_categorical(batch_size * num_episodes, pi_A_2, pi_B_A))
        for i in range(num_episodes):
            x_transfer = all_x_transfer[:(batch_size * (i + 1))]

            val_loss_A_B = -torch.mean(model.model_A_B(x_transfer))
            val_loss_B_A = -torch.mean(model.model_B_A(x_transfer))

            grad_norm_A_B = get_gradient_norm(model, val_loss_A_B)
            grad_norm_B_A = get_gradient_norm(model, val_loss_B_A)

            record = grad_norm_A_B - grad_norm_B_A

            losses[:, k, j, i] = [record, 0]

flat_losses = -losses.reshape((2, -1, num_episodes))
losses_25, losses_50, losses_75 = np.percentile(flat_losses, (25, 50, 75), axis=1)

plt.figure(figsize=(18, 12))

ax = plt.subplot(1, 1, 1)
ax.plot(losses_50[0], color='C0', label=r'difference', lw=6)
ax.fill_between(np.arange(num_episodes), losses_25[0], losses_75[0], color='C0', alpha=0.2)
ax.plot(losses_50[1], color='C3', label=r'zero', lw=6)
ax.fill_between(np.arange(num_episodes), losses_25[1], losses_75[1], color='C3', alpha=0.2)
ax.set_xlim([0, flat_losses.shape[1] - 1])
ax.tick_params(axis='both', which='major', labelsize=36)
ax.set_xlabel(r'Number of examples ($\times$100)', fontsize=40)
ax.set_ylabel(r'$\mathcal{S}_{L_2}$', fontsize=40)

plt.show()