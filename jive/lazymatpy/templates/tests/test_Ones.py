
import numpy as np

from jive.lazymatpy.templates.ones import Ones, OnesOuterVec
from jive.lazymatpy.templates.tests.help_fun import get_tst_mats

from numpy.testing import assert_allclose


class TestOnes(object):

    def test_basic(self):
        # shape = (10, 5)

        shapes = [(10, 5), (1, 5), (10, 1)]
        for shape in shapes:

            A = Ones(shape)
            M = np.ones(shape)

            V = np.random.normal(size=(shape[1], 3))
            v1 = np.random.normal(size=(shape[1], ))
            v2 = np.random.normal(size=(shape[1], 1))
            
            U = np.random.normal(size=(shape[0], 3))
            u1 = np.random.normal(size=(shape[0], ))
            u2 = np.random.normal(size=(shape[0], 1))

            assert_allclose(A.dot(V), M.dot(V))
            assert_allclose(A.dot(v1), M.dot(v1))
            assert_allclose(A.dot(v2), M.dot(v2))

            assert_allclose(A.T.dot(U), M.T.dot(U))
            assert_allclose(A.T.dot(u1), M.T.dot(u1))
            assert_allclose(A.T.dot(u2), M.T.dot(u2))


class TestOnesOuterVec(object):

    def test_basic(self):
        vecs = [[1, 2, 3],
                [2],
                np.array([1, 2, 3]).reshape(1, -1),
                np.array([1, 2, 3]).reshape(-1, 1)]

        num_ones_list = [4, 1]

        for vec in vecs:
            for num_ones in num_ones_list:

                A = OnesOuterVec(num_ones=num_ones, vec=vec)
                M = np.outer([1]*num_ones, vec)

                V, v1, v2, U, u1, u2 = get_tst_mats(M.shape)

                assert_allclose(A.dot(V), M.dot(V))
                assert_allclose(A.dot(v1), M.dot(v1))
                assert_allclose(A.dot(v2), M.dot(v2))

                assert_allclose(A.T.dot(U), M.T.dot(U))
                assert_allclose(A.T.dot(u1), M.T.dot(u1))
                assert_allclose(A.T.dot(u2), M.T.dot(u2))










