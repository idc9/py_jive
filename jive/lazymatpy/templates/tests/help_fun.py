import numpy as np

def get_tst_mats(shape, ncol=3, seed=None):
    """
    Returns test matrices/vectors for a matrix M.

    M.dot(V)
    M.dot(v1)
    M.dot(v2)

    M.T.dot(U)
    M.T.dot(u1)
    M.T.dot(u2)

    V/U are matrices, v1/u1 are arrays with one dimension (e.g. shape (5,))
    and v2/u2 are arrays with two dimensions (e.g. shape (5, 1))

    Parameters
    ----------
    shape: the shape of M
    
    ncol: number of columns for U/V matrices
    
    seed: if given will generate random matrices

    Output
    ------
    V, v1, v2, U, u1, u2
    """

    if len(shape) == 1:
        shape = (shape[0], 1)

    if seed is None:
        nrow = shape[1]
        V = np.arange(1,  ncol * nrow + 1).reshape((nrow, ncol))
        v1 = np.arange(1, nrow + 1)
        v2 = v1.reshape(-1, 1)

        nrow = shape[0]
        U = np.arange(1,  ncol * nrow + 1).reshape((nrow, ncol))
        u1 = np.arange(1, nrow + 1)
        u2 = u1.reshape(-1, 1)

    else:
        # TODO: set seed
        V = np.random.normal(size=(shape[1], ncol))
        v1 = np.random.normal(size=(shape[1], ))
        v2 = np.random.normal(size=(shape[1], 1))
        
        U = np.random.normal(size=(shape[0], ncol))
        u1 = np.random.normal(size=(shape[0], ))
        u2 = np.random.normal(size=(shape[0], 1))

    return V, v1, v2, U, u1, u2


# def mat_mul_test(A, M, d=3, seed=None):

#     shape = A.shape

#     V = np.random.normal(size=(shape[1], d))
#     v1 = np.random.normal(size=(shape[1], ))
#     v2 = np.random.normal(size=(shape[1], 1))
    
#     U = np.random.normal(size=(shape[0], d))
#     u1 = np.random.normal(size=(shape[0], ))
#     u2 = np.random.normal(size=(shape[0], 1))

#     assert_allclose(A.dot(V), M.dot(V))
#     assert_allclose(A.dot(v1), M.dot(v1))
#     assert_allclose(A.dot(v2), M.dot(v2))

#     assert_allclose(A.T.dot(U), M.T.dot(U))
#     assert_allclose(A.T.dot(u1), M.T.dot(u1))
#     assert_allclose(A.T.dot(u2), M.T.dot(u2))