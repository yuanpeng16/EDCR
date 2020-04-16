import argparse
import random
from sklearn.linear_model import LinearRegression


class Model(object):
    def prepare_data(self, data):
        X, y = list(zip(*data))
        X = [[a] for a in X]
        return X, y

    def train(self, data):
        X, y = self.prepare_data(data)
        self.reg = LinearRegression().fit(X, y)
        return self.reg.coef_, self.reg.intercept_

    def test(self, data):
        X, y = self.prepare_data(data)
        y_hat = self.reg.predict(X)
        assert len(y) == len(y_hat)
        loss = sum([(a - b) ** 2 for a, b in zip(y, y_hat)]) / (2 * len(y))
        return loss


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


def get_generalization_loss(data_train, data_transfer):
    model = Model()
    model.train(data_train)
    loss_train = model.test(data_train)
    loss_transfer = model.test(data_transfer)
    generalization_loss = loss_transfer - loss_train
    return generalization_loss


def experiment(data):
    # prepare data
    data_train_AB, data_transfer_AB = separate_distributions(data)
    data_train_BA = [[x[1], x[0]] for x in data_train_AB]
    data_transfer_BA = [[x[1], x[0]] for x in data_transfer_AB]

    # compute losses and results
    gen_loss_AB = get_generalization_loss(data_train_AB, data_transfer_AB)
    gen_loss_BA = get_generalization_loss(data_train_BA, data_transfer_BA)
    score = gen_loss_BA - gen_loss_AB
    result = score > 0
    return 1 if result else 0


def main(args):
    # load data
    with open(args.file_name, 'r') as f:
        lines = f.readlines()
    data = [[float(x) for x in line.strip().split(' ')] for line in lines]

    # run experiments
    success = 0
    for _ in range(args.experiments):
        success += experiment(data)
    success_rate = (100.0 * success) / args.experiments
    print('Success rate is', success_rate, '% (', success, 'out of',
          args.experiments, 'are correct )')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compositional Instructions.')
    parser.add_argument('--file_name', type=str, default='pair0001.txt',
                        help='input file')
    parser.add_argument('--experiments', type=int, default=1000,
                        help='number of experiments')
    args = parser.parse_args()
    main(args)
