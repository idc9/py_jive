
import numpy as np
from scipy.sparse import rand


from jive.lazymatpy.templates.matrix_transformations import col_mean_centered, col_proj, col_proj_orthog
from jive.lazymatpy.templates.tests.help_fun import get_tst_mats

from numpy.testing import assert_allclose, assert_almost_equal






class Test_col_mean_centered(object):

    def get_Xs(self, shape, seed=None):

        X_sparse = rand(shape[0], shape[1])
        X_ndarray = X_sparse.toarray()
        X_mat = X_sparse.todense()

        return X_sparse, X_ndarray, X_mat

    def test_basic(self):

        shapes = [(50, 20), (1, 20), (50, 1)]

        # sparse
        for shape in shapes:
            mats = self.get_Xs(shape)

            m = mats[0].mean(axis=0).A1
            ones = np.ones(shape[0])
            M = mats[0].toarray() - np.outer(ones, m)

            for X in mats:

                A = col_mean_centered(X)

                V, v1, v2, U, u1, u2 = get_tst_mats(M.shape)

                assert_almost_equal(A.dot(V), M.dot(V))
                assert_almost_equal(A.dot(v1), M.dot(v1))
                assert_almost_equal(A.dot(v2), M.dot(v2))

                assert_almost_equal(A.T.dot(U), M.T.dot(U))
                assert_almost_equal(A.T.dot(u1), M.T.dot(u1))
                assert_almost_equal(A.T.dot(u2), M.T.dot(u2))

        
class Test_col_proj(object):

    def test_basic(self):

        shape = (50, 20)
        X = rand(shape[0], shape[1])
        U = np.random.normal(size=(shape[0], 15))
        M = np.dot(np.dot(U, U.T), X.toarray())

        A = col_proj(X, U)

        V, v1, v2, U, u1, u2 = get_tst_mats(M.shape)

        assert_almost_equal(A.dot(V), M.dot(V))
        assert_almost_equal(A.dot(v1), M.dot(v1))
        assert_almost_equal(A.dot(v2), M.dot(v2))

        assert_almost_equal(A.T.dot(U), M.T.dot(U))
        assert_almost_equal(A.T.dot(u1), M.T.dot(u1))
        assert_almost_equal(A.T.dot(u2), M.T.dot(u2))


class Test_col_proj_orthog(object):

    def test_basic(self):

        shape = (50, 20)
        X = rand(shape[0], shape[1])
        U = np.random.normal(size=(shape[0], 15))
        M = np.dot(np.eye(shape[0]) - np.dot(U, U.T), X.toarray())

        A = col_proj_orthog(X, U)

        V, v1, v2, U, u1, u2 = get_tst_mats(M.shape)

        assert_almost_equal(A.dot(V), M.dot(V))
        assert_almost_equal(A.dot(v1), M.dot(v1))
        assert_almost_equal(A.dot(v2), M.dot(v2))

        assert_almost_equal(A.T.dot(U), M.T.dot(U))
        assert_almost_equal(A.T.dot(u1), M.T.dot(u1))
        assert_almost_equal(A.T.dot(u2), M.T.dot(u2))



