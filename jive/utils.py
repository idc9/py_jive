import numpy as np
from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd

from jive.lazymatpy.interface import LinearOperator
from jive.lazymatpy.convert2scipy import convert2scipy


def svd_wrapper(X, rank=None):
    """
    Computes the (possibly partial) SVD of a matrix. Handles the case where
    X is either dense or sparse.

    Parameters
    ----------
    X: either dense or sparse
    rank: rank of the desired SVD (required for sparse matrices)

    Output
    ------
    U, D, V
    the columns of U are the left singular vectors
    the COLUMNS of V are the left singular vectors

    """
    if isinstance(X, LinearOperator):
        scipy_svds = svds(convert2scipy(X), rank)
        U, D, V = fix_scipy_svds(scipy_svds)

    elif issparse(X) or rank is not None:
        scipy_svds = svds(X, rank)
        U, D, V = fix_scipy_svds(scipy_svds)

    else:
        # TODO: implement partial SVD
        U, D, V = full_svd(X, full_matrices=False)
        V = V.T

        if rank:
            U = U[:, :rank]
            D = D[:rank]
            V = V[:, :rank]

    return U, D, V


def fix_scipy_svds(scipy_svds):
    """
    scipy.sparse.linalg.svds orders the singular values backwards,
    this function fixes this insanity and returns the singular values
    in decreasing order

    Parameters
    ----------
    scipy_svds: the out put from scipy.sparse.linalg.svds

    Output
    ------
    U, D, V
    ordered in decreasing singular values
    """
    U, D, V = scipy_svds

    sv_reordering = np.argsort(-D)

    U = U[:, sv_reordering]
    D = D[sv_reordering]
    V = V.T[:, sv_reordering]

    return U, D, V


def centering(X, method='mean'):

    if method == 'mean':
        center = np.array(X.mean(axis=0)).reshape(-1)
    else:
        center = np.zeros(X.shape[1])

    if issparse(X):
        raise NotImplementedError
        # return MeanCentered(blocks[bn], centers_[bn]), center

    else:
        return X - center, center


def pca_wrapper(X, rank=None, center_method='mean'):
    """
    PCA = mean center then SVD
    """
    return svd_wrapper(centering(X, method=center_method)[0], rank=rank)
