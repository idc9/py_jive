import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns


def plot_data_blocks(blocks):
    """
    Plots a heat map of a bunch of data blocks
    """
    num_blocks = len(blocks)

    plt.figure(figsize=[5 * num_blocks,  5])

    for k in range(num_blocks):

        plt.subplot(1, num_blocks, k + 1)
        sns.heatmap(blocks[k], xticklabels=False, yticklabels=False)
        plt.title('block ' + str(k))


def plot_jive_full_estimates(full_block_estimates, blocks):
    """
    Plots the full JVIE estimates: X, J, I, E
    """
    num_blocks = len(full_block_estimates)

    plt.figure(figsize=[10, num_blocks * 10])

    for k in range(num_blocks):

        # grab data
        X = blocks[k]
        J = full_block_estimates[k]['J']
        I = full_block_estimates[k]['I']
        E = full_block_estimates[k]['E']

        # observed data
        plt.subplot(4, num_blocks, k + 1)
        sns.heatmap(X, xticklabels=False, yticklabels=False)
        plt.title('block ' + str(k) + ' observed data')

        # full joint estimate
        plt.subplot(4, num_blocks, k + num_blocks + 1)
        sns.heatmap(J, xticklabels=False, yticklabels=False)
        plt.title('joint')

        # full individual estimate
        plt.subplot(4, num_blocks, k + 2 * num_blocks + 1)
        sns.heatmap(I, xticklabels=False, yticklabels=False)
        plt.title('individual')

        # full noise estimate
        plt.subplot(4, num_blocks, k + 3 * num_blocks + 1)
        sns.heatmap(E, xticklabels=False, yticklabels=False)
        plt.title('noise ')
