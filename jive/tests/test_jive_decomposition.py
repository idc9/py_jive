import unittest
import numpy as np

from jive.lin_alg_fun import *
from jive.ajive_fig2 import *
from jive.Jive import *


class JiveDecomposition(unittest.TestCase):
    """
    Make sure all JIVE constraints work.
    """
    def setUp(self):
        """
        Sample data and compute JIVE estimates
        """

        # sample platonic data
        seed = 23423
        X_obs, X_joint, X_indiv, X_noise, Y_obs, Y_joint, Y_indiv, Y_noise = generate_data_ajive_fig2(seed)

        blocks = [X_obs, Y_obs]
        jive = Jive(blocks)
        jive.compute_initial_svd()
        jive.set_signal_ranks([2, 3])
        jive.compute_joint_svd()
        jive.estimate_jive_spaces()

        self.block_specific_estimates = jive.get_block_specific_estimates()
        self.common_joint_space_estimate = jive.get_common_joint_space_estimate()

        self.blocks=blocks
        self.K = len(self.blocks)
        self.dimensions = [blocks[k].shape[1] for k in range(self.K)]
        self.joint_rank = 1
        self.individual_ranks = [1, 2]


    def test_jive_decoposition(self):
        """
        Check that X = I + J + E for each block
        """
        for k in range(self.K):
            X = self.blocks[k]

            J = self.block_specific_estimates[k]["joint"]["full"]
            I = self.block_specific_estimates[k]["individual"]["full"]
            E = self.block_specific_estimates[k]["noise"]

            residual = X - (J + I + E)
            self.assertTrue(np.allclose(residual, 0))

    def test_joint_rank(self):
        """
        Check JIVE has the correct joint rank
        """
        # TODO: this might be a stochastic test -- maybe romeves
        rank_estimate = self.block_specific_estimates[0]['joint']['rank']

        self.assertTrue(self.joint_rank == rank_estimate)

    def test_individual_ranks(self):
        """
        Check JIVE has the correct individual rank
        """
        for k in range(self.K):
            rank_estimate = self.block_specific_estimates[k]['individual']['rank']
            rank_true = self.individual_ranks[k]

            self.assertTrue(rank_true == rank_estimate)

    def test_individual_space_SVD(self):
        """
        Check the SVD of the I matrix is correct
        """
        for k in range(self.K):
            I = self.block_specific_estimates[k]["individual"]["full"]
            U = self.block_specific_estimates[k]["individual"]["scores"]
            D = self.block_specific_estimates[k]["individual"]["sing_vals"]
            V = self.block_specific_estimates[k]["individual"]["loadings"]

            # compare SVD reconstruction to I matrix
            svd_reconstruction = np.dot(U, np.dot(np.diag(D), V.T))
            residual = I - svd_reconstruction

            self.assertTrue(np.allclose(residual, 0))

    def test_block_joint_space_SVD(self):
        """
        Check the SVD of the J matrix is correct
        """
        for k in range(self.K):
            J = self.block_specific_estimates[k]["joint"]["full"]
            U = self.block_specific_estimates[k]["joint"]["scores"]
            D = self.block_specific_estimates[k]["joint"]["sing_vals"]
            V = self.block_specific_estimates[k]["joint"]["loadings"]

            # compare SVD reconstruction to I matrix
            svd_reconstruction = np.dot(U, np.dot(np.diag(D), V.T))
            residual = J - svd_reconstruction

            self.assertTrue(np.allclose(residual, 0))

    def test_joint_space_svd(self):
        """
        Check the joint space.
        """
        # TODO: come up with more tests
        U = self.common_joint_space_estimate['scores']
        D = self.common_joint_space_estimate['sing_vals']
        V = self.common_joint_space_estimate['loadings']
        rank = self.common_joint_space_estimate['rank']

        # check shapes
        self.assertTrue(U.shape[1] == rank)
        self.assertTrue(V.shape[1] == rank)
        self.assertTrue(len(D) == rank)

        # make sure the joint ranks are consistent
        self.assertTrue(self.block_specific_estimates[0]["joint"]["rank"] ==
                        self.common_joint_space_estimate['rank'])

        # TODO: compare joint space reconstruction to something


if __name__ == '__main__':
    unittest.main()
