import os
import numpy as np

def write_data(directory, filename, a_list, b_list, c_list):
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, filename)
    with open(filename, 'w') as f:
        for i, (a, b, c) in enumerate(zip(a_list, b_list, c_list)):
            f.write(str(i + 1))
            f.write('\t')
            f.write(str(a))
            f.write('\t')
            f.write(str(b))
            f.write('\t')
            f.write(str(c))
            f.write('\n')

def read_data(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
    ret = []
    for line in lines:
        terms = line.strip().split('\t')
        ret.append([float(term) for term in terms])
    ret = np.transpose(np.asarray(ret))
    return ret