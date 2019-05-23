import sys
sys.path.insert(0, '../..')

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm_notebook

from causal_meta.utils.data_utils import generate_data_categorical
from proposed_models import StructuralModel

parser = argparse.ArgumentParser(description='Compositional Instructions.')
parser.add_argument('--N', type=int, default=10, help='num units')
args = parser.parse_args()

# fix data
np.random.seed(4)

N = args.N
model = StructuralModel(N, dtype=torch.float64)

num_episodes = 100
batch_size = 100 # 1
num_test = 10000
num_training = 10 # 100
num_transfers = 10 # 100

losses = np.zeros((2, num_training, num_transfers, num_episodes))

for k in tnrange(num_training):
    pi_A_1 = np.random.dirichlet(np.ones(N))
    pi_B_A = np.random.dirichlet(np.ones(N), size=N)

    model.set_ground_truth(pi_A_1, pi_B_A)
    x_train_original = torch.from_numpy(generate_data_categorical(
        num_test, pi_A_1, pi_B_A))
    with torch.no_grad():
        original_loss_A_B = -torch.mean(model.model_A_B(x_train_original))
        original_loss_B_A = -torch.mean(model.model_B_A(x_train_original))
        original_loss_A_B = original_loss_A_B.item()
        original_loss_B_A = original_loss_B_A.item()

    for j in tnrange(num_transfers, leave=False):
        pi_A_2 = np.random.dirichlet(np.ones(N))
        all_x_transfer = torch.from_numpy(generate_data_categorical(batch_size * num_episodes, pi_A_2, pi_B_A))
        for i in range(num_episodes):
            x_transfer = all_x_transfer[:(batch_size * (i + 1))]
            with torch.no_grad():
                inner_loss_A_B = -torch.mean(model.model_A_B(x_transfer))
                inner_loss_B_A = -torch.mean(model.model_B_A(x_transfer))
                loss_A_B = inner_loss_A_B.item() - original_loss_A_B
                loss_B_A = inner_loss_B_A.item() - original_loss_B_A

            losses[:, k, j, i] = [loss_A_B - loss_B_A, 0]


flat_losses = -losses.reshape((2, -1, num_episodes))
losses_25, losses_50, losses_75 = np.percentile(flat_losses, (25, 50, 75), axis=1)

plt.figure(figsize=(18, 12))

ax = plt.subplot(1, 1, 1)
ax.plot(losses_50[0], color='C0', label='Score', lw=6)
ax.fill_between(np.arange(num_episodes), losses_25[0], losses_75[0], color='C0', alpha=0.2)
ax.plot(losses_50[1], color='C3', label='Zero', lw=6)
ax.fill_between(np.arange(num_episodes), losses_25[1], losses_75[1], color='C3', alpha=0.2)
ax.set_xlim([0, flat_losses.shape[1] - 1])
ax.tick_params(axis='both', which='major', labelsize=36)
ax.legend(loc=4, prop={'size': 36})
ax.set_xlabel(r'Number of examples ($\times$100)', fontsize=40)
ax.set_ylabel(r'$\mathcal{S}_\mathcal{G}}$', fontsize=40)

plt.show()