import matplotlib.pyplot as plt
from utils import read_data

def plot(proposed, baseline, num_transfer, y_label, file_name, lines):
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1)

    ax.tick_params(axis='both', which='major', labelsize=36)
    for line in lines:
        ax.axhline(line, lw=6, c='lightgray', ls='--', zorder=0)
    ax.plot(proposed, lw=6, color='green', zorder=2)
    ax.plot(baseline, lw=6, color='orange', zorder=1)

    ax.set_xlim([0, num_transfer - 1])
    ax.set_xlabel('Number of episodes', fontsize=48)
    ax.set_ylabel(y_label, fontsize=48)
    plt.savefig(file_name)

def main():
    baseline = read_data("results/baseline_N=100.txt")
    proposed = read_data("results/proposed_meta_N=100.txt")
    plot(proposed[1], baseline[1], len(baseline[0]), '$\sigma(\gamma)$', "results/direction_gamma_N=100.pdf", [0, 0.5, 1])
    plot(proposed[2] * 100, baseline[2] * 100, 150, 'Accuracy (%)', "results/direction_acc_N=100.pdf", [0, 50, 100])

if __name__ == '__main__':
    main()