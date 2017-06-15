import unittest
import numpy as np

from jive.jive.lin_alg_fun import *
from jive.jive.ajive_fig2 import *
from jive.jive.jive import *


class JiveOptions(unittest.TestCase):
    """
    Make sure different JIVE options run without error.
    """
    def setUp(self):
        """
        Sample data.
        """
        # sample platonic data
        seed = 23423
        X_obs, X_joint, X_indiv, X_noise, \
            Y_obs, Y_joint, Y_indiv, Y_noise = generate_data_ajive_fig2(seed)

        self.blocks = [X_obs, Y_obs]

    def test_different_options(self):
        blocks = self.blocks

        jive = Jive(blocks, wedin_estimate=True, full=True,
                    show_scree_plot=True)
        jive.set_signal_ranks([2, 3])
        block_estimates = jive.get_block_estimates()
        self.assertTrue(True)

        jive = Jive(blocks, wedin_estimate=True, full=True,
                    show_scree_plot=False)
        jive.set_signal_ranks([2, 3])
        block_estimates = jive.get_block_estimates()
        self.assertTrue(True)

        jive = Jive(blocks, wedin_estimate=False, full=True,
                    show_scree_plot=False)
        jive.set_signal_ranks([2, 3])
        jive.set_joint_rank(joint_rank=1)
        block_estimates = jive.get_block_estimates()
        self.assertTrue(True)

        jive = Jive(blocks, wedin_estimate=False, full=False,
                    show_scree_plot=False)
        jive.set_signal_ranks([2, 3])
        jive.set_joint_rank(joint_rank=1)
        block_estimates = jive.get_block_estimates()
        self.assertTrue(True)

    def test_full_true(self):
        """
        Make sure that full matrices are returned in the block estimates.
        Make sure we can compute the full block estimates.
        """
        jive = Jive(self.blocks, wedin_estimate=False, full=True,
                    show_scree_plot=False)
        jive.set_signal_ranks([2, 3])
        jive.set_joint_rank(joint_rank=1)
        full_block_estimates = jive.get_block_estimates_full()
        block_estimates = jive.get_block_estimates()

        for k in range(len(self.blocks)):
            X = self.blocks[k]

            J = block_estimates[k]["joint"]["full"]
            I = block_estimates[k]["individual"]["full"]
            E = block_estimates[k]["noise"]

            # check all J, I, E are in the block_estimats
            residual = X - (J + I + E)
            self.assertTrue(np.allclose(residual, 0))


            #  make sure we can compute the full block estimates
            X = self.blocks[k]
            J = full_block_estimates[k]['J']
            I = full_block_estimates[k]['I']
            E = full_block_estimates[k]['E']

            residual = X - (J + I + E)
            self.assertTrue(np.allclose(residual, 0))

    def test_full_false(self):
        """
        Make sure full matrices are not automatically comptued
        when full = False. Make sure we can compute the full block estimates.
        """
        jive = Jive(self.blocks, wedin_estimate=False, full=False,
                    show_scree_plot=False)
        jive.set_signal_ranks([2, 3])
        jive.set_joint_rank(joint_rank=1)
        block_estimates = jive.get_block_estimates()
        full_block_estimates = jive.get_block_estimates_full()

        for k in range(len(self.blocks)):

            J = block_estimates[k]["joint"]["full"]
            I = block_estimates[k]["individual"]["full"]
            E = block_estimates[k]["noise"]

            # make sure J, I, E are all empty
            self.assertTrue(J.shape == (1, 0))
            self.assertTrue(I.shape == (1, 0))
            self.assertTrue(E.shape == (1, 0))

            X = self.blocks[k]
            J = full_block_estimates[k]['J']
            I = full_block_estimates[k]['I']
            E = full_block_estimates[k]['E']

            # make sure we can compute the full block estimates
            residual = X - (J + I + E)
            self.assertTrue(np.allclose(residual, 0))


if __name__ == '__main__':
    unittest.main()
