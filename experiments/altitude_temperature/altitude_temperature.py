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
        X, Y = list(zip(*data))
        Y = list(zip(*Y))
        return X, Y

    def train(self, data):
        X, Y = self.prepare_data(data)
        self.reg = [self.get_model().fit(X, y) for y in Y]

    def test(self, data):
        X, Y = self.prepare_data(data)
        return np.mean([1 - reg.score(X, y) for reg, y in zip(self.reg, Y)])


class NNModel(Model):
    def get_model(self):
        hidden_layer_sizes = [100] * self.args.hidden_layers
        return MLPRegressor(hidden_layer_sizes=hidden_layer_sizes,
                            max_iter=1000)


def separate_distributions(data):
    random.shuffle(data)
    data.sort(key=lambda x: x[0][0])
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
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j] = np.random.normal(data[i][j], noise_std)
    return data


def experiment(args, data, noise_std):
    # prepare data
    data_train_AB, data_transfer_AB = separate_distributions(data)
    data_train_AB = np.asarray(data_train_AB)
    data_transfer_AB = np.asarray(data_transfer_AB)
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


def is_reverse(file_name):
    if len(file_name) < 5 or not file_name.startswith('data'):
        return False
    key = file_name.split('/')[1][:8]
    with open('data/README', 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith(key):
            return '<-' in line
    assert False


def scale_variable(variable):
    v_list = np.transpose(variable)
    v_list = [preprocessing.scale(v) for v in v_list]
    variable = np.transpose(v_list)
    return variable


def read_float(x):
    if x == 'NaN':
        return 0
    return float(x)


def get_special_datasets(lines, file_name):
    if len(file_name) < 5 or not file_name.startswith('data'):
        return False
    key = file_name.split('/')[1][:8]
    m = {
        'pair0071': [6, 2, True],
        'pair0081': [1, 1, False],
        'pair0082': [1, 1, False],
        'pair0083': [1, 1, False],
        'pair0105': [9, 1, True],
    }
    if key in m:
        a, b, is_multidimensional = m[key]
        data = [[read_float(x) for x in line.strip().split()] for line in
                lines if len(line.strip()) > 0]
        data = [[x[:a], x[a:a + b]] for x in data]
        return data, is_multidimensional

    is_multidimensional = '\t' in lines[0]
    if is_multidimensional:
        data = [[[read_float(y) for y in x.split()] for x in
                 line.strip().split('\t')] for line in lines if
                len(line.strip()) > 0]
    else:
        data = [[[read_float(x)] for x in line.strip().split(' ')] for line in
                lines if len(line.strip()) > 0]
    return data, is_multidimensional


def load_data(args):
    with open(args.file_name, 'r') as f:
        lines = f.readlines()
    data, is_multidimensional = get_special_datasets(lines, args.file_name)

    if is_reverse(args.file_name):
        data = np.flip(data, axis=1)
    if args.balance_scale or args.scale != 1:
        X, Y = list(zip(*data))
        if args.balance_scale:
            X = scale_variable(X)
            Y = scale_variable(Y)
        if args.scale != 1:
            Y *= args.scale
        data = list(zip(X, Y))
    data = np.asarray(data)
    return data.tolist(), is_multidimensional


def main(args):
    # load data
    data, is_multidimensional = load_data(args)

    # run experiments
    success = 0
    for _ in range(args.experiments):
        success += experiment(args, data, args.noise)
    success_rate = (100.0 * success) / args.experiments
    multi = 'multidimensional' if is_multidimensional else 'scalar'
    print('Success rate is', success_rate, '% (', success, 'out of',
          args.experiments, 'are correct )', multi)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compositional Instructions.')
    parser.add_argument('--file_name', type=str,
                        default='data/pair0052.txt',
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
