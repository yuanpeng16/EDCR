import matplotlib.pyplot as plt
from utils import read_data

def plot(proposed, baseline, num_transfer, y_label, file_name):
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1)

    ax.tick_params(axis='both', which='major', labelsize=36)
    ax.axhline(0.5, lw=6, c='lightgray', ls='--', zorder=0)
    ax.axhline(-0.5, lw=6, c='lightgray', ls='--', zorder=0)
    ax.plot(proposed, lw=6, color='green', zorder=2)
    ax.plot(baseline, lw=6, color='orange', zorder=1)

    ax.set_xlim([0, num_transfer - 1])
    ax.set_xlabel('Iterations', fontsize=48)
    ax.set_ylabel(y_label, fontsize=48)
    plt.savefig(file_name)

def main():
    baseline = read_data("results/baseline.txt")
    proposed = read_data("results/proposed.txt")
    plot(proposed[1], baseline[1], min(800, len(baseline[0])), 'Encoder Angle [$\pi$/2 rad]', "results/representation_sample_gpu.pdf")

if __name__ == '__main__':
    main()