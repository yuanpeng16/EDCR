import matplotlib.pyplot as plt
from utils import read_data

def plot(proposed_acc, proposed_time, baseline_acc, baseline_time, num_transfer, y_label, file_name):
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1)

    ax.tick_params(axis='both', which='major', labelsize=36)
    ax.axhline(0.5, lw=6, c='lightgray', ls='--', zorder=0)
    ax.axhline(-0.5, lw=6, c='lightgray', ls='--', zorder=0)

    ax.scatter(proposed_time[:num_transfer], proposed_acc[:num_transfer], color='green', zorder=2)
    ax.scatter(baseline_time[:num_transfer], baseline_acc[:num_transfer], color='orange', zorder=1)

    ax.set_xlabel('Time (sec)', fontsize=48)
    ax.set_ylabel(y_label, fontsize=48)
    plt.savefig(file_name)

def main():
    baseline = read_data("results/baseline.txt")
    proposed = read_data("results/proposed.txt")
    plot(proposed[1], proposed[2], baseline[1], baseline[2], min(800, len(baseline[0])), 'Encoder Angle [$\pi$/2 rad]', "results/representation_time_gpu.pdf")

if __name__ == '__main__':
    main()