'''
@Project ：Improved physics-informed neural network in mitigating gradient-related failur 
@File    ：plot_loss.py
@IDE     ：PyCharm 
@Author  ：Pancheng Niu
@Date    ：2024/7/18 下午3:58 
'''
import os
import csv
import matplotlib.pyplot as plt
import math


def read_loss_csv(file_path, sample_interval=1):
    """Read loss values from a CSV file and sample every sample_interval"""
    iterations = []
    losses = []
    with open(file_path, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for i, row in enumerate(reader):
            if i % sample_interval == 0:
                iterations.append(int(row[0]))
                losses.append(float(row[1]))
    return iterations, losses


def plot_losses(directory, mode, layer_str, lam_threshold_list, sample_interval=1):
    """Read all loss logs and plot the graph."""
    plt.figure()
    for lam_threshold in lam_threshold_list:
        file_path = os.path.join(directory, f'{mode}_{layer_str}_loss_log_lam_threshold_{lam_threshold}.csv')
        if os.path.exists(file_path):
            iterations, losses = read_loss_csv(file_path, sample_interval)
            exponent = int(math.log10(lam_threshold))
            plt.plot(iterations, losses, label=f'$\gamma$= $10^{exponent}$')
        else:
            print(f'File not found: {file_path}')

    plt.yscale('log')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'{mode}_{layer_str}_Loss_vs_Iterations_sampled.pdf'), dpi=300,
                bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    directory = './pic'
    mode = 'I_PINN'
    layer_str = '7x50'
    lam_threshold_list = [1, 10, 100, 1000, 10000, 100000, 1000000]
    sample_interval = 10

    plot_losses(directory, mode, layer_str, lam_threshold_list, sample_interval)