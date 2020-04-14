import matplotlib.pyplot as plt
import numpy as np

def plot_iter(proposed, baseline, num_transfer, file_name):
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1)

    ax.tick_params(axis='both', which='major', labelsize=36)
    ax.axhline(0.5, lw=6, c='lightgray', ls='--', zorder=0)
    ax.axhline(-0.5, lw=6, c='lightgray', ls='--', zorder=0)
    ax.plot(proposed[:,0], lw=6, color='green', zorder=2)
    ax.plot(baseline[:,0], lw=6, color='orange', zorder=1)

    ax.set_xlim([0, num_transfer - 1])
    ax.set_xlabel('Iterations', fontsize=48)
    ax.set_ylabel('Encoder Angle [$\pi$/2 rad]', fontsize=48)
    plt.savefig(file_name)

def plot_time(proposed, baseline, num_transfer, file_name):
    fig = plt.figure(figsize=(18, 12))
    ax = plt.subplot(1, 1, 1)

    ax.tick_params(axis='both', which='major', labelsize=36)
    ax.axhline(0.5, lw=6, c='lightgray', ls='--', zorder=0)
    ax.axhline(-0.5, lw=6, c='lightgray', ls='--', zorder=0)

    ax.scatter(proposed[:num_transfer,1], proposed[:num_transfer,0], color='green', zorder=2)
    ax.scatter(baseline[:num_transfer,1], baseline[:num_transfer,0], color='orange', zorder=1)

    ax.set_xlabel('Time (sec)', fontsize=48)
    ax.set_ylabel('Encoder Angle [$\pi$/2 rad]', fontsize=48)
    plt.savefig(file_name)

def main():
    baseline = np.load("results/baseline.npy")
    baseline = np.mean(baseline, axis=0)
    baseline[:,0] = baseline[:,0] / (np.pi / 2)

    proposed = np.load("results/proposed.npy")
    proposed = np.mena(proposed, axis=0)
    proposed[:,0] = proposed[:,0] / (np.pi / 2)

    num_transfer = min(800, len(baseline[:,0]))
    plot_iter(proposed, baseline, num_transfer, "results/representation_runs_iter.pdf")
    plot_time(proposed, baseline, num_transfer, "results/representation_runs_time.pdf")

if __name__ == '__main__':
    main()
