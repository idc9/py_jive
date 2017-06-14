import unittest
import numpy as np

from help_fun import *
from jive.jive.ajive_fig2 import *
from jive.jive.jive import *


class AJIVEFig2(unittest.TestCase):
    """AJIVE figure 2 example"""
    def setUp(self):
        """
        Sample data and compute JIVE estimates
        """

        # sample platonic data
        seed = 23423
        self.X_obs, self.X_joint, self.X_indiv, self.X_noise, \
        self.Y_obs, self.Y_joint, self.Y_indiv, self. Y_noise = generate_data_ajive_fig2(seed)

        # compute JIVE decomposition
        jive = Jive([self.X_obs, self.Y_obs])
        jive.set_signal_ranks([2, 3])  # we know the true ranks
        self.jive_estimates = jive.get_jive_estimates()

        self.rel_err_tolerance = .5

    def test_individual_rank_estimates(self):
        """
        Check that JIVE found the correct indivudal space rank estimate.
        """

        x_indiv_rank_est = self.jive_estimates[0]["individual_rank"]
        y_indiv_rank_est = self.jive_estimates[1]["individual_rank"]

        self.assertTrue(x_indiv_rank_est == 1)
        self.assertTrue(y_indiv_rank_est == 2)

    def test_joint_rank_estimates(self):
        joint_rank_est = self.jive_estimates['joint_rank']
        self.assertTrue(joint_rank_est == 1)

    def test_relative_error_x(self):
        J_est = self.jive_estimates[0]["J"]
        I_est = self.jive_estimates[0]["I"]
        E_est = self.jive_estimates[0]["E"]

        joint_rel_err = relative_error(self.X_joint, J_est)
        indiv_rel_err = relative_error(self.X_indiv, I_est)
        noise_rel_err = relative_error(self.X_noise, E_est)

        self.assertTrue(joint_rel_err < self.rel_err_tolerance)
        self.assertTrue(indiv_rel_err < self.rel_err_tolerance)
        self.assertTrue(noise_rel_err < self.rel_err_tolerance)

    def test_relative_error_y(self):
        J_est = self.jive_estimates[1]["J"]
        I_est = self.jive_estimates[1]["I"]
        E_est = self.jive_estimates[1]["E"]

        joint_rel_err = relative_error(self.Y_joint, J_est)
        indiv_rel_err = relative_error(self.Y_indiv, I_est)
        noise_rel_err = relative_error(self.Y_noise, E_est)

        self.assertTrue(joint_rel_err < self.rel_err_tolerance)
        self.assertTrue(indiv_rel_err < self.rel_err_tolerance)
        self.assertTrue(noise_rel_err < self.rel_err_tolerance)


if __name__ == '__main__':
    unittest.main()
