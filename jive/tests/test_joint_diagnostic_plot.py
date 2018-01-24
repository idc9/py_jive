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

        # compute JIVE decomposition
        jive = Jive(blocks=blocks)

        jive.compute_initial_svd()
        jive.set_signal_ranks([2, 3])
        jive.compute_joint_svd()
        jive.estimate_joint_rank()

        self.jive = jive


    def test_joint_diagnostic_plot(self):
        """
        Make sure plot_joint_diagnostic() runs without error.
        """
        self.jive.plot_joint_diagnostic()
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
