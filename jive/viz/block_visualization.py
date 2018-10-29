import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def data_block_heatmaps(blocks):
    """
    Plots a heat map of a bunch of data blocks
    """
    num_blocks = len(blocks)
    for k in range(num_blocks):
        plt.subplot(1, num_blocks, k + 1)
        sns.heatmap(blocks[k], xticklabels=False, yticklabels=False, cmap='RdBu')
        plt.title('block ' + str(k))


def jive_full_estimate_heatmaps(full_block_estimates, blocks):
    """
    Plots the full JVIE estimates: X, J, I, E
    """
    num_blocks = len(full_block_estimates)

    # plt.figure(figsize=[10, num_blocks * 10])

    for k, bn in enumerate(full_block_estimates.keys()):

        # grab data
        X = blocks[bn]
        J = full_block_estimates[bn]['joint']
        I = full_block_estimates[bn]['individual']
        E = full_block_estimates[bn]['noise']

        # observed data
        plt.subplot(4, num_blocks, k + 1)
        sns.heatmap(X, xticklabels=False, yticklabels=False, cmap='RdBu')
        plt.title('block ' + str(k) + ' observed data')

        # full joint estimate
        plt.subplot(4, num_blocks, k + num_blocks + 1)
        sns.heatmap(J, xticklabels=False, yticklabels=False, cmap='RdBu')
        plt.title('joint')

        # full individual estimate
        plt.subplot(4, num_blocks, k + 2 * num_blocks + 1)
        sns.heatmap(I, xticklabels=False, yticklabels=False, cmap='RdBu')
        plt.title('individual')

        # full noise estimate
        plt.subplot(4, num_blocks, k + 3 * num_blocks + 1)
        sns.heatmap(E, xticklabels=False, yticklabels=False, cmap='RdBu')
        plt.title('noise ')
