import argparse
import random
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn import preprocessing


class Model(object):
    def __init__(self, args):
        self.args = args

    def get_model(self):
        return LinearRegression()

    def prepare_data(self, data):
        X, y = list(zip(*data))
        X = [[a] for a in X]
        return X, y

    def train(self, data):
        X, y = self.prepare_data(data)
        self.reg = self.get_model().fit(X, y)

    def test(self, data):
        X, y = self.prepare_data(data)
        return 1 - self.reg.score(X, y)


class NNModel(Model):
    def get_model(self):
        hidden_layer_sizes = [100] * self.args.hidden_layers
        return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                            max_iter=1000)


def separate_distributions(data):
    random.shuffle(data)
    data.sort(key=lambda x: x[0])
    thresh = len(data) // 2
    left, right = data[thresh:], data[:thresh]

    random.shuffle(left)
    random.shuffle(right)

    thresh_left = len(left) // 10
    left_1, left_2 = left[thresh_left:], left[:thresh_left]
    thresh_right = len(right) // 10
    right_1, right_2 = right[thresh_right:], right[:thresh_right]

    data_train = left_1 + right_2
    data_transfer = right_1 + left_2
    return data_train, data_transfer


def get_generalization_loss(args, data_train, data_transfer):
    if args.hidden_layers == 0:
        model = Model(args)
    else:
        model = NNModel(args)
    model.train(data_train)
    loss_train = model.test(data_train)
    loss_transfer = model.test(data_transfer)
    generalization_loss = loss_transfer - loss_train
    return generalization_loss


def add_noise(data, noise_std):
    data = np.asarray(data)
    data = np.random.normal(data, noise_std)
    return data


def experiment(args, data, noise_std):
    # prepare data
    data_train_AB, data_transfer_AB = separate_distributions(data)
    if noise_std > 0:
        data_train_AB = add_noise(data_train_AB, noise_std)
        data_transfer_AB = add_noise(data_transfer_AB, noise_std)
    data_train_BA = [[x[1], x[0]] for x in data_train_AB]
    data_transfer_BA = [[x[1], x[0]] for x in data_transfer_AB]

    # compute losses and results
    gen_loss_AB = get_generalization_loss(args, data_train_AB,
                                          data_transfer_AB)
    gen_loss_BA = get_generalization_loss(args, data_train_BA,
                                          data_transfer_BA)
    score = gen_loss_BA - gen_loss_AB
    result = score > 0
    return 1 if result else 0


def main(args):
    # load data
    with open(args.file_name, 'r') as f:
        lines = f.readlines()
    data = [[float(x) for x in line.strip().split(' ')] for line in lines]
    if args.balance_scale or args.scale != 1:
        X, Y = list(zip(*data))
        if args.balance_scale:
            X = preprocessing.scale(X)
            Y = preprocessing.scale(Y)
        if args.scale != 1:
            Y *= args.scale
        data = list(zip(X, Y))

    # run experiments
    success = 0
    for _ in range(args.experiments):
        success += experiment(args, data, args.noise)
    success_rate = (100.0 * success) / args.experiments
    print('Success rate is', success_rate, '% (', success, 'out of',
          args.experiments, 'are correct )')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compositional Instructions.')
    parser.add_argument('--file_name', type=str, default='pair0001.txt',
                        help='input file')
    parser.add_argument('--experiments', type=int, default=1000,
                        help='number of experiments')
    parser.add_argument('--balance_scale', action='store_true', default=False,
                        help='scale data')
    parser.add_argument('--scale', type=float, default=1,
                        help='scaling')
    parser.add_argument('--noise', type=float, default=0,
                        help='std of noise')
    parser.add_argument('--hidden_layers', type=int, default=0,
                        help='number of hidden layers')
    args = parser.parse_args()
    main(args)
