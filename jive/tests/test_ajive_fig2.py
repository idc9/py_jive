import unittest
import numpy as np

from jive.lin_alg_fun import *
from jive.ajive_fig2 import *
from jive.Jive import *


class AjiveFig2(unittest.TestCase):
    """
    AJIVE figure 2 example

    Check if JIVE can find the correct signals.
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

        blocks = [self.X_obs, self.Y_obs]
        jive = Jive(blocks)
        jive.compute_initial_svd()
        jive.set_signal_ranks([2, 3])
        jive.compute_joint_svd()
        jive.estimate_jive_spaces()

        self.block_estimates = jive.block_specific_estimates = jive.get_block_specific_estimates()

    def test_individual_rank_estimates(self):
        """
        Check that JIVE found the correct indivudal space rank estimate.
        """
        x_indiv_rank_est = self.block_estimates[0]["individual"]['rank']
        y_indiv_rank_est = self.block_estimates[1]["individual"]['rank']

        self.assertTrue(x_indiv_rank_est == 1)
        self.assertTrue(y_indiv_rank_est == 2)

    def test_joint_rank_estimates(self):
        joint_rank_est = self.block_estimates[0]["joint"]['rank']
        self.assertTrue(joint_rank_est == 1)

    def test_relative_error_x(self):
        J_est = self.block_estimates[0]["joint"]["full"]
        I_est = self.block_estimates[0]["individual"]["full"]
        E_est = self.block_estimates[0]["noise"]

        joint_rel_err = relative_error(self.X_joint, J_est)
        indiv_rel_err = relative_error(self.X_indiv, I_est)
        noise_rel_err = relative_error(self.X_noise, E_est)

        self.assertTrue(joint_rel_err < self.rel_err_tolerance)
        self.assertTrue(indiv_rel_err < self.rel_err_tolerance)
        self.assertTrue(noise_rel_err < self.rel_err_tolerance)

    def test_relative_error_y(self):
        J_est = self.block_estimates[1]["joint"]["full"]
        I_est = self.block_estimates[1]["individual"]["full"]
        E_est = self.block_estimates[1]["noise"]

        joint_rel_err = relative_error(self.Y_joint, J_est)
        indiv_rel_err = relative_error(self.Y_indiv, I_est)
        noise_rel_err = relative_error(self.Y_noise, E_est)

        self.assertTrue(joint_rel_err < self.rel_err_tolerance)
        self.assertTrue(indiv_rel_err < self.rel_err_tolerance)
        self.assertTrue(noise_rel_err < self.rel_err_tolerance)


if __name__ == '__main__':
    unittest.main()
