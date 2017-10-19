import unittest
import numpy as np

from jive.ajive_fig2 import *
from jive.Jive import *
from jive.jive_visualization import *


class JiveViz(unittest.TestCase):
    """
    Make sure visualization fucntions run

    """
    def setUp(self):
        """
        Sample data and compute JIVE estimates
        """
        # sample platonic data
        seed = 23423
        X_obs, X_joint, X_indiv, X_noise, \
        Y_obs, Y_joint, Y_indiv, Y_noise = generate_data_ajive_fig2(seed)

        blocks = [X_obs, Y_obs]
        init_svd_ranks = None
        wedin_estimate = True
        save_full_final_decomp = True

        # compute JIVE decomposition
        self.jive = Jive(blocks=blocks,
                         init_svd_ranks=init_svd_ranks,
                         wedin_estimate=wedin_estimate,
                         save_full_final_decomp=save_full_final_decomp)


        self.jive.set_signal_ranks([2, 3])  # we know the true ranks

        self.blocks = blocks

    def test_block_plot(self):
        """
        Make sure plot_data_blocks() runs without error.
        """
        plot_data_blocks(self.blocks)
        self.assertTrue(True)

    def test_jive_plot(self):
        """
        Make sure plot_jive_full_estimates() runs without error.
        """
        plot_jive_full_estimates(self.jive)
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
