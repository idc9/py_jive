import unittest
import numpy as np

from jive.jive.ajive_fig2 import *
from jive.jive.jive import *
from jive.jive.jive_visualization import *


class JiveViz(unittest.TestCase):
    """
    Make sure visualization fucntions run

    """
    def setUp(self):
        """
        Sample data and compute JIVE estimates
        """
        # TODO: think more about this number currently artibrary
        self.rel_err_tolerance = .5

        # sample platonic data
        seed = 23423
        self.X_obs, self.X_joint, self.X_indiv, self.X_noise, \
        self.Y_obs, self.Y_joint, self.Y_indiv, self. Y_noise = generate_data_ajive_fig2(seed)

        self.blocks = [self.X_obs, self.Y_obs]
        wedin_bound = True
        show_scree_plot = False
        full = True

        # compute JIVE decomposition
        self.jive = Jive(self.blocks , wedin_bound, full, show_scree_plot)
        self.jive.set_signal_ranks([2, 3])  # we know the true ranks

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
