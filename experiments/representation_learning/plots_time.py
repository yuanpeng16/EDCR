import matplotlib.pyplot as plt
from utils import read_data

def plot(proposed_acc, proposed_time, baseline_acc, baseline_time, num_transfer, y_label, file_name):
    fig = plt.figure(figsize=(9, 5))
    ax = plt.subplot(1, 1, 1)

    ax.tick_params(axis='both', which='major', labelsize=13)
    ax.axhline(0.5, c='lightgray', ls='--')
    ax.axhline(-0.5, c='lightgray', ls='--')

    ax.scatter(proposed_time[:num_transfer], proposed_acc[:num_transfer], color='green')
    ax.scatter(baseline_time[:num_transfer], baseline_acc[:num_transfer], color='orange')


    ax.set_xlabel('Time (sec)', fontsize=14)
    ax.set_ylabel(y_label, fontsize=14)
    plt.savefig(file_name)

def main():
    baseline = read_data("results/baseline.txt")
    proposed = read_data("results/proposed.txt")
    plot(proposed[1], proposed[2], baseline[1], baseline[2], min(800, len(baseline[0])), 'Encoder Angle [$\pi$/2 rad]', "results/theta_time.pdf")

if __name__ == '__main__':
    main()