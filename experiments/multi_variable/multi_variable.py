import sys
sys.path.insert(0, '../..')

import argparse
import time
import numpy as np
import torch
import random

from causal_meta.utils.data_utils import generate_data_categorical
from multi_variable_models import StructuralModel
from utils import write_data

parser = argparse.ArgumentParser()
parser.add_argument('--proposed', action='store_true', default=False,
                    help='use proposed method')
parser.add_argument('--m', type=int, default=10, help='num variabless')
parser.add_argument('--N', type=int, default=10, help='num units')
parser.add_argument('--max_adaptation_steps', type=int, default=500, help='maximum number of steps in adaptation.')
args = parser.parse_args()

# fix random seeds
torch.manual_seed(4)
np.random.seed(4)

m = args.m
N = args.N
model = StructuralModel(N, dtype=torch.float64)

meta_optimizer = torch.optim.RMSprop([model.w], lr=1e-2)

repeats = 5
num_runs = (m * (m - 1)) // 2
num_training = 1
num_transfer = args.max_adaptation_steps
num_gradient_steps = 2

train_batch_size = 1000
transfer_batch_size = 10

alphas = np.zeros((num_runs, num_training, num_transfer))
accs = np.zeros((num_runs, num_training, num_transfer))
times = np.zeros((num_runs, num_training, num_transfer))

count = 0
for r in range(repeats):
    print(r, '/', repeats)
    directions = []
    for i in range(m):
        for j in range(i):
            directions.append([i, j, random.randint(0, 1)])

    valid = True
    for j in range(num_runs):
        direction = directions[j]
        if direction[2] == 0:
            direction = direction[1], direction[0], 1 - direction[2]

        model.w.data.zero_()
        for i in range(num_training):
            # Step 1: Sample a joint distribution before intervention
            pi_A_1 = np.random.dirichlet(np.ones(N))
            pi_B_A = np.random.dirichlet(np.ones(N), size=N)

            model.set_ground_truth(pi_A_1, pi_B_A)
            x_train_original = torch.from_numpy(generate_data_categorical(
                train_batch_size, pi_A_1, pi_B_A))
            with torch.no_grad():
                original_loss_A_B = -torch.mean(model.model_A_B(x_train_original))
                original_loss_B_A = -torch.mean(model.model_B_A(x_train_original))

            cum_AtoB = 0
            cum_BtoA = 0
            transfers = range(num_transfer)
            for k in transfers:
                # Step 2: Train the modules on the training distribution
                model.set_ground_truth(pi_A_1, pi_B_A)

                # Step 3: Sample a joint distribution after intervention
                pi_A_2 = np.random.dirichlet(np.ones(N))

                start = time.time()
                # Step 4: Do k steps of gradient descent for adaptation on the
                # distribution after intervention
                model.zero_grad()
                loss = torch.tensor(0., dtype=torch.float64)
                for _ in range(num_gradient_steps):
                    x_train = torch.from_numpy(generate_data_categorical(transfer_batch_size, pi_A_2, pi_B_A))
                    with torch.no_grad():
                        loss_A_B = model.model_A_B(x_train) + original_loss_A_B
                        loss_B_A = model.model_B_A(x_train) + original_loss_B_A
                        cum_AtoB += -torch.mean(loss_A_B)
                        cum_BtoA += -torch.mean(loss_B_A)
                    loss += -torch.mean(model.online_loglikelihood(loss_A_B, loss_B_A))

                # Step 5: Update the structural parameter alpha
                meta_optimizer.zero_grad()
                loss.backward()
                meta_optimizer.step()
                end = time.time()

                # Log the values of alpha
                with torch.no_grad():
                    alpha = torch.sigmoid(model.w).item()

                if cum_AtoB < cum_BtoA:
                    predicted = 1.
                else:
                    predicted = 0.

                alphas[j, i, k] = alpha
                accs[j, i, k] = predicted
                times[j, i, k] = end - start
                if k > 0:
                    times[j, i, k] += times[j, i, k - 1]
        if predicted != direction[2]:
            valid = False
    if valid:
        count += 1

success_rate = (100.0 * count) / repeats
print('Success rate:', success_rate, '% (', count, 'out of', repeats, 'are correct)')

alpha_mean = np.mean(alphas.reshape((-1, num_transfer)), axis=0)
acc_mean = np.mean(accs.reshape((-1, num_transfer)), axis=0)
time_mean = np.mean(times.reshape((-1, num_transfer)), axis=0)

write_data("results", "proposed_meta_N=" + str(N) + ".txt", alpha_mean, acc_mean, time_mean)
