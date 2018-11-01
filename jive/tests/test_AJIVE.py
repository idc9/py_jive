import unittest

import numpy as np
from jive.AJIVE import AJIVE
from jive.ajive_fig2 import generate_data_ajive_fig2
from jive.tests.utils import svd_checker


class TestFig2Runs(unittest.TestCase):

    @classmethod
    def setUp(self):
        X, Y = generate_data_ajive_fig2()
        ajive = AJIVE(init_signal_ranks={'x': 2, 'y': 3})
        ajive.fit(blocks={'x': X, 'y': Y})
        self.ajive = ajive
        self.X = X
        self.Y = Y

    def test_has_attributes(self):
        """
        Check AJIVE has important attributes
        """
        self.assertTrue(hasattr(self.ajive, 'blocks'))
        self.assertTrue(hasattr(self.ajive, 'common'))
        self.assertTrue(hasattr(self.ajive.blocks['x'], 'joint'))
        self.assertTrue(hasattr(self.ajive.blocks['x'], 'individual'))
        self.assertTrue(hasattr(self.ajive.blocks['y'], 'joint'))
        self.assertTrue(hasattr(self.ajive.blocks['y'], 'individual'))

    def test_correct_estimates(self):
        """
        Check AJIVE found correct rank estimates
        """
        self.assertEqual(self.ajive.common.rank, 1)
        self.assertEqual(self.ajive.blocks['x'].individual.rank, 1)
        self.assertEqual(self.ajive.blocks['y'].individual.rank, 2)

    def test_matrix_decomposition(self):
        """
        check X = I + J + E
        """
        Rx = self.X - (self.ajive.blocks['x'].joint.full_ +
                       self.ajive.blocks['x'].individual.full_ +
                       self.ajive.blocks['x'].noise_)

        self.assertTrue(np.allclose(Rx, 0))

        Ry = self.Y - (self.ajive.blocks['y'].joint.full_ +
                       self.ajive.blocks['y'].individual.full_ +
                       self.ajive.blocks['y'].noise_)

        self.assertTrue(np.allclose(Ry, 0))

    def test_diagnostic_plot(self):
        """
        Check the diagnostic plot runs
        """
        self.ajive.plot_joint_diagnostic()

    def test_common_estimates(self):
        """
        Check common SVD
        """
        U, D, V = self.ajive.common.get_UDV()
        rank = self.ajive.common.rank
        n = self.X.shape[0]
        d = sum(self.ajive.init_signal_ranks.values())
        self.assertTrue(svd_checker(U, D, V, n, d, rank))

    def test_block_specific_estimates(self):
        U, D, V = self.ajive.blocks['x'].joint.get_UDV()
        rank = 1
        n, d = self.X.shape
        self.assertTrue(svd_checker(U, D, V, n, d, rank))

        U, D, V = self.ajive.blocks['x'].individual.get_UDV()
        rank = 1
        n, d = self.X.shape
        self.assertTrue(svd_checker(U, D, V, n, d, rank))

        U, D, V = self.ajive.blocks['y'].joint.get_UDV()
        rank = 1
        n, d = self.X.shape
        self.assertTrue(svd_checker(U, D, V, n, d, rank))

        U, D, V = self.ajive.blocks['y'].individual.get_UDV()
        rank = 2
        n, d = self.X.shape
        self.assertTrue(svd_checker(U, D, V, n, d, rank))


if __name__ == '__main__':
    unittest.main()
