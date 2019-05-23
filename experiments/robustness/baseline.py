import sys
sys.path.insert(0, '../..')

import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tnrange, tqdm_notebook

from causal_meta.utils.data_utils import generate_data_categorical
from models import StructuralModel

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

optimizer = torch.optim.SGD(model.modules_parameters(), lr=1.)

losses = np.zeros((2, num_training, num_transfers, num_episodes))

for k in tnrange(num_training):
    pi_A_1 = np.random.dirichlet(np.ones(N))
    pi_B_A = np.random.dirichlet(np.ones(N), size=N)
    for j in tnrange(num_transfers, leave=False):
        model.set_ground_truth(pi_A_1, pi_B_A)

        # train P(A)
        a_model = model.model_A_B.p_A
        a_optimizer = torch.optim.Adam(a_model.parameters(), lr=0.1)
        for i in range(num_episodes):
            x_train = torch.from_numpy(generate_data_categorical(batch_size, pi_A_1, pi_B_A))
            inputs_A, inputs_B = torch.split(x_train, 1, dim=1)
            a_model.zero_grad()
            a_loss = -torch.mean(a_model(inputs_A))
            a_loss.backward()
            a_optimizer.step()

        pi_A_2 = np.random.dirichlet(np.ones(N))
        x_val = torch.from_numpy(generate_data_categorical(num_test, pi_A_2, pi_B_A))
        for i in range(num_episodes):
            x_transfer = torch.from_numpy(generate_data_categorical(batch_size, pi_A_2, pi_B_A))
            model.zero_grad()
            loss_A_B = -torch.mean(model.model_A_B(x_transfer))
            loss_B_A = -torch.mean(model.model_B_A(x_transfer))
            loss = loss_A_B + loss_B_A

            with torch.no_grad():
                val_loss_A_B = -torch.mean(model.model_A_B(x_val))
                val_loss_B_A = -torch.mean(model.model_B_A(x_val))

            losses[:, k, j, i] = [val_loss_A_B.item(), val_loss_B_A.item()]

            loss.backward()
            optimizer.step()

flat_losses = -losses.reshape((2, -1, num_episodes))
losses_25, losses_50, losses_75 = np.percentile(flat_losses, (25, 50, 75), axis=1)

plt.figure(figsize=(18, 12))

ax = plt.subplot(1, 1, 1)
ax.plot(losses_50[0], color='C0', label=r'$A \rightarrow B$', lw=6)
ax.fill_between(np.arange(num_episodes), losses_25[0], losses_75[0], color='C0', alpha=0.2)
ax.plot(losses_50[1], color='C3', label=r'$B \rightarrow A$', lw=6)
ax.fill_between(np.arange(num_episodes), losses_25[1], losses_75[1], color='C3', alpha=0.2)
ax.set_xlim([0, flat_losses.shape[1] - 1])
ax.tick_params(axis='both', which='major', labelsize=36)
ax.legend(loc=4, prop={'size': 36})
ax.set_xlabel(r'Number of examples ($\times$100)', fontsize=40)
ax.set_ylabel(r'$\log P(D\mid \cdot \rightarrow \cdot)$', fontsize=40)

plt.show()