import unittest

import numpy as np
import pandas as pd
from jive.PCA import PCA
from jive.tests.utils import svd_checker


class TestPCA(unittest.TestCase):

    @classmethod
    def setUp(self):
        self.n = 100
        self.d = 20
        self.n_components = 10
        self.obs_names = ['sample_{}'.format(i) for i in range(self.n)]
        self.var_names = ['var_{}'.format(i) for i in range(self.d)]
        X = np.random.normal(size=(self.n, self.d))
        self.X = pd.DataFrame(X, index=self.obs_names, columns=self.var_names)
        self.pca = PCA(n_components=self.n_components).fit(self.X)

    def test_has_attributes(self):
        """
        Check AJIVE has important attributes
        """
        self.assertTrue(hasattr(self.pca, 'scores_'))
        self.assertTrue(hasattr(self.pca, 'svals_'))
        self.assertTrue(hasattr(self.pca, 'loadings_'))
        self.assertTrue(hasattr(self.pca, 'm_'))
        self.assertTrue(hasattr(self.pca, 'frob_norm_'))

    def test_name_extraction(self):
        self.assertTrue(set(self.pca.obs_names()), set(self.obs_names))
        self.assertTrue(set(self.pca.var_names()), set(self.var_names))

    def test_shapes(self):
        self.assertEqual(self.pca.shape_, (self.n, self.d))

        self.assertEqual(self.pca.scores_.shape, (self.n, self.n_components))
        self.assertEqual(self.pca.scores().shape, (self.n, self.n_components))
        self.assertEqual(self.pca.scores(norm=True).shape, (self.n, self.n_components))
        self.assertEqual(self.pca.scores(norm=False).shape, (self.n, self.n_components))

        self.assertEqual(self.pca.loadings_.shape, (self.d, self.n_components))
        self.assertEqual(self.pca.loadings().shape, (self.d, self.n_components))

        self.assertEqual(self.pca.svals_.shape, (self.n_components, ))
        self.assertEqual(self.pca.svals().shape, (self.n_components, ))

    def test_pca_formatting(self):
        U, D, V = self.pca.get_UDV()
        n, d = self.X.shape
        rank = self.n_components
        self.assertTrue(svd_checker(U, D, V, n, d, rank))

    def test_plots(self):
        self.pca.plot_loading(comp=0)
        self.pca.plot_scores_hist(comp=1)
        self.pca.plot_scree()
        self.pca.plot_var_expl_prop()
        self.pca.plot_var_expl_cum()
        self.pca.plot_scores()
        self.pca.plot_scores_vs(comp=1, y=np.random.normal(size=self.X.shape[0]))
        # self.pca.plot_interactive_scores_slice(1, 3)

    def test_frob_norm(self):
        """
        Check Frobenius norm is calculated correctly whether the full
        or partial PCA is computed.
        """
        true_frob_norm = np.linalg.norm(self.X, ord='fro')
        pca = PCA(n_components=None).fit(self.X)
        self.assertTrue(np.allclose(pca.frob_norm_, true_frob_norm))

        pca = PCA(n_components=3).fit(self.X)
        self.assertTrue(np.allclose(pca.frob_norm_, true_frob_norm))
