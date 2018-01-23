import numpy as np

from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd

import matplotlib.pyplot as plt

from jive.lazymatpy.interface import LinearOperator
from jive.lazymatpy.convert2scipy import convert2scipy
from jive.lazymatpy.templates.matrix_transformations import svd_residual

def svd_wrapper(X, rank = None):
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

    elif issparse(X):
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

def svds_additional(A, U_first, D_first, V_first, k):
    """
    Computes the next k SVD components given the first several SVD componenets.

    Parametrs
    ---------
    A: data matrix
    U_first, D_first, V_first: first several SVD compoents of A
    k: how many additional SVD componenets to compute
    """

    R = svd_residual(A, U_first, D_first, V_first.T)

    U, D, V = svd_wrapper(R, k)

    # TODO: finish this
    U = np.block([U_first, U])
    D = np.block([D_first, D])
    V = np.block([V_first, V])

    return U, D, V

def scree_plot(sv, log=False, diff=False, title=''):
    """
    Makes a scree plot
    """
    ylab = 'singular value'

    # possibly log values
    if log:
        sv = np.log(sv)
        ylab = 'log ' + ylab

    # possibly take differences
    if diff:
        sv = np.diff(sv)
        ylab = ylab + ' difference'

    n = len(sv)

    plt.scatter(range(n), sv)
    plt.plot(range(n), sv)
    plt.ylim([1.1*min(0, min(sv)), 1.1*max(sv)])
    plt.xlim([-.01 * n, n])
    plt.xticks(int(n/10.0) * np.arange(10))
    plt.xlabel('index')
    plt.ylabel(ylab)
    plt.title(title)


def relative_error(true_value, estimate):
    """
    Relative error under L2/frobenius norm
    """
    # TODO: what if true value is zero?
    return np.linalg.norm(true_value - estimate) / np.linalg.norm(true_value)


def absolute_error(true_value, estimate):
    """
    Absolute error under L2/frobenius norm
    """
    return np.linalg.norm(true_value - estimate)
