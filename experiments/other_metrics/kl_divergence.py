import sys
sys.path.insert(0, '../..')

#%matplotlib inline
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
batch_size = 1 # 1
num_test = 10000
num_training = 10 # 100
num_transfers = 10 # 100

#optimizer = torch.optim.SGD(model.modules_parameters(), lr=1.)

losses = np.zeros((2, num_training, num_transfers, num_episodes))

for k in tnrange(num_training):
    pi_A_1 = np.random.dirichlet(np.ones(N))
    pi_B_A = np.random.dirichlet(np.ones(N), size=N)
    for j in tnrange(num_transfers, leave=False):
        model.set_ground_truth(pi_A_1, pi_B_A)
        pi_A_2 = np.random.dirichlet(np.ones(N))
        x_val = torch.from_numpy(generate_data_categorical(num_test, pi_A_2, pi_B_A))

        #transfer_model.set_ground_truth(pi_A_2, pi_B_A)

        all_x_transfer = torch.from_numpy(generate_data_categorical(batch_size * num_episodes, pi_A_2, pi_B_A))
        for i in range(num_episodes):
            x_transfer = all_x_transfer[:(batch_size * (i + 1))]

            transfer_model = StructuralModel(N, dtype=torch.float64)
            transfer_model.set_maximum_likelihood(x_transfer)
            #optimizer = torch.optim.SGD(transfer_model.modules_parameters(), lr=1.)
            #for l in range(100):
            #    model.zero_grad()
            #    loss_A_B = -torch.mean(transfer_model.model_A_B(x_transfer))
            #    loss_B_A = -torch.mean(transfer_model.model_B_A(x_transfer))
            #    loss = loss_A_B + loss_B_A
            #    loss.backward()
            #    optimizer.step()
            #    print(loss)
            #    if loss.item() == 0:
            #        break

            with torch.no_grad():
                val_loss_A_B = -torch.mean(model.model_A_B(x_transfer))
                val_loss_B_A = -torch.mean(model.model_B_A(x_transfer))
                val_loss_A_B += torch.mean(transfer_model.model_A_B(x_transfer))
                val_loss_B_A += torch.mean(transfer_model.model_B_A(x_transfer))

            losses[:, k, j, i] = [val_loss_A_B - val_loss_B_A, 0]

#            loss.backward()
#            optimizer.step()

flat_losses = -losses.reshape((2, -1, num_episodes))
losses_25, losses_50, losses_75 = np.percentile(flat_losses, (25, 50, 75), axis=1)

plt.figure(figsize=(9, 5))

ax = plt.subplot(1, 1, 1)
ax.plot(losses_50[0], color='C0', lw=2)
ax.fill_between(np.arange(num_episodes), losses_25[0], losses_75[0], color='C0', alpha=0.2)
ax.plot(losses_50[1], color='C3', lw=2)
ax.fill_between(np.arange(num_episodes), losses_25[1], losses_75[1], color='C3', alpha=0.2)
ax.set_xlim([0, flat_losses.shape[1] - 1])
ax.tick_params(axis='both', which='major', labelsize=13)
#ax.legend(loc=4, prop={'size': 13})
ax.set_xlabel('Number of transfer examples', fontsize=14)
ax.set_ylabel(r'$\Delta D_{KL}$', fontsize=14)

plt.show()