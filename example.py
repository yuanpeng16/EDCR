import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from matplotlib import rc


# Data preparation
def normalize(P):
    return P / np.sum(P)


class DataGenerator(object):
    def get_conditional_distribution(self, N_a, N_b):
        P_ab = normalize(np.random.rand(N_a, N_b))
        P_a = np.sum(P_ab, -1)
        P_b_a = P_ab / np.expand_dims(P_a, -1)
        return P_b_a

    def get_train_marginal_distribution(self, N_a):
        return normalize(np.random.rand(N_a))

    def get_transfer_marginal_distribution(self, N_a):
        return self.get_train_marginal_distribution(N_a)

    def get_joint_distribution(self, P_a, P_b_a):
        P_ab = np.expand_dims(P_a, -1) * P_b_a
        assert np.isclose(np.sum(P_ab), 1.0)
        return P_ab

    def get_data(self, N_a, N_b):
        P_b_a = self.get_conditional_distribution(N_a, N_b)
        assert np.all(np.isclose(np.sum(P_b_a, -1), np.ones(N_a)))

        P1_a = self.get_train_marginal_distribution(N_a)
        assert np.isclose(np.sum(P1_a), 1.0)

        P2_a = self.get_transfer_marginal_distribution(N_a)
        assert np.isclose(np.sum(P2_a), 1.0)

        P1_ab = self.get_joint_distribution(P1_a, P_b_a)
        P2_ab = self.get_joint_distribution(P2_a, P_b_a)
        return P1_ab, P2_ab


class DeterministicDataGenerator(DataGenerator):
    def get_conditional_distribution(self, N_a, N_b):
        return np.asarray([[0.3, 0.7], [0.6, 0.4]])

    def get_train_marginal_distribution(self, N_a):
        return np.asarray([0.4, 0.6])

    def get_transfer_marginal_distribution(self, N_a):
        return np.asarray([0.2, 0.8])


# Computation
def entropy(P):
    return - np.sum(P * np.log(P))


def marginal_entropy(P_ab, a=True):
    axis = 1 if a else 0
    return entropy(np.sum(P_ab, axis))


def conditional_distribution(P_xy):
    '''Compute conditional distribution.
    Input P_xy: P(XY) in (Rx x Ry).
    return P(Y|X) in (Rx x Ry). Each row sums to 1.'''
    assert len(P_xy.shape) == 2
    axis = 1
    P_x = np.sum(P_xy, axis)
    P_y_x = P_xy / np.expand_dims(P_x, axis)
    return P_y_x


def kl_divergence(P_xy, Q_xy):
    '''D_KL(P(y|x),Q(y|x)) = sum P(xy) (log P(y|x) - log Q(y|x))'''
    P_y_x = conditional_distribution(P_xy)
    Q_y_x = conditional_distribution(Q_xy)
    kl = Q_xy * (np.log(Q_y_x) - np.log(P_y_x))
    kl = np.sum(kl)
    return kl


def compute_score(P1_ab, P2_ab):
    H1_a = marginal_entropy(P1_ab, True)
    H2_a = marginal_entropy(P2_ab, True)
    DH_a = H2_a - H1_a

    H1_b = marginal_entropy(P1_ab, False)
    H2_b = marginal_entropy(P2_ab, False)
    DH_b = H2_b - H1_b
    D = DH_b - DH_a

    P1_ba = np.transpose(P1_ab)
    P2_ba = np.transpose(P2_ab)
    D_ba = kl_divergence(P1_ba, P2_ba)
    S = D_ba - D
    return S, [D_ba, DH_b, DH_a]


# Direct computation
def cross_entropy(P_xy, Q_xy):
    Q_y_x = conditional_distribution(Q_xy)
    return -np.sum(P_xy * np.log(Q_y_x))


def gap(P1_ab, P2_ab):
    XE_ab = cross_entropy(P2_ab, P1_ab)
    H = cross_entropy(P1_ab, P1_ab)
    G = XE_ab - H
    return G


def get_direct_score(P1_ab, P2_ab):
    G_ab = gap(P1_ab, P2_ab)
    P1_ba = np.transpose(P1_ab)
    P2_ba = np.transpose(P2_ab)
    G_ba = gap(P1_ba, P2_ba)
    S = G_ba - G_ab
    return S


def experiment_with_data(P1_ab, P2_ab):
    R = get_direct_score(P1_ab, P2_ab)
    S, T = compute_score(P1_ab, P2_ab)
    assert np.isclose(S, R)
    return S, T


def experiment(N_a, N_b):
    dg = DataGenerator()
    P1_ab, P2_ab = dg.get_data(N_a, N_b)
    return experiment_with_data(P1_ab, P2_ab)


def one_plot(x, y, filename):
    plt.axhline(0, lw=3, c='lightgray', ls='--', zorder=0)
    plt.axvline(0, lw=3, c='lightgray', ls='--', zorder=0)
    plt.scatter(x, y, s=5, c='black')
    plt.ylabel('$\mathcal{S}_\mathcal{G}$')
    plt.savefig(filename, bbox_inches='tight', pad_inches=0.01)
    plt.clf()


def plot_Ha(x, y, folder, variable_size):
    plt.xlabel('$\Delta H(A)$')
    one_plot(x, y, os.path.join(folder, 'dHa_' + str(variable_size) + '.pdf'))


def plot_Dkl(x, y, folder, variable_size):
    plt.axline([0, 0], slope=1, lw=3, c='lightgray', ls='--', zorder=0)
    plt.xlabel('$\mathcal{S}_{D_{KL}}$')
    one_plot(x, y, os.path.join(folder, 'D_kl_' + str(variable_size) + '.pdf'))


def plot(matrix, variable_size):
    # S, D_ba, DH_b, DH_a
    font = {'family': 'serif', 'size': 14}
    rc('font', **font)

    folder = 'example_results'
    os.makedirs(folder, exist_ok=True)

    matrix = np.transpose(np.asarray(matrix))
    y = matrix[0]
    plot_Dkl(matrix[1], y, folder, variable_size)
    plot_Ha(matrix[3], y, folder, variable_size)


def randomized_experiments(args):
    np.random.seed(args.data_random_seed)
    N_a = args.variable_size  # rows
    N_b = args.variable_size  # columns

    num = args.num_experiments
    hit = 0
    results = []
    for _ in range(num):
        S, T = experiment(N_a, N_b)
        if S > 0:
            hit += 1
        results.append([S] + T)
    print(hit / num, hit)
    plot(results, args.variable_size)


def example_experiments(args):
    dg = DeterministicDataGenerator()
    P1_ab, P2_ab = dg.get_data(2, 2)
    S, _ = experiment_with_data(P1_ab, P2_ab)
    print("S =", S)


def main(args):
    if args.experiment_type == 'example':
        example_experiments(args)
    else:
        randomized_experiments(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_type', type=str, default='example',
                        help='Experiment type.')
    parser.add_argument('--data_random_seed', type=int, default=8,
                        help='Random seed.')
    parser.add_argument('--variable_size', type=int, default=10,
                        help='Size of a variable.')
    parser.add_argument('--num_experiments', type=int, default=1000,
                        help='Number of experiments.')
    main(parser.parse_args())
