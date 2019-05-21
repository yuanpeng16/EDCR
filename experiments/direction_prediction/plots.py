import matplotlib.pyplot as plt
from utils import read_data

def plot(proposed, baseline, num_transfer, y_label, file_name):
    fig = plt.figure(figsize=(9, 5))
    ax = plt.subplot(1, 1, 1)

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.axhline(1, c='lightgray', ls='--')
    ax.axhline(0.5, c='lightgray', ls='--')
    ax.axhline(0, c='lightgray', ls='--')
    ax.plot(proposed, lw=2, color='green')
    ax.plot(baseline, lw=2, color='orange')

    ax.set_xlim([0, num_transfer - 1])
    ax.set_xlabel('Number of episodes', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    plt.savefig(file_name)

def main():
    baseline = read_data("results/baseline_N=10.txt")
    proposed = read_data("results/proposed_meta_N=10.txt")
    plot(proposed[1], baseline[1], len(baseline[0]), '$\sigma(\gamma)$', "results/direction_gamma_N=10.pdf")
    plot(proposed[2], baseline[2], 50, 'Accuracy', "results/direction_acc_N=10.pdf")

if __name__ == '__main__':
    main()