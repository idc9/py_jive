import numpy as np
import matplotlib.pyplot as plt


def projection_matrix(X):
    """
    Returns the projection matrix of X

    P = X(X^T X)^{-1}X^T

    i.e. Pv projects v onto the columns of X
    """
    if len(X.shape) == 1:  # x is a vector
        return np.outer(X, X)
    else:
        return np.dot(np.dot(X, np.linalg.inv(np.dot(X.T, X))), X.T)

# TODO: kill this
# def orthogonal_projection_matrix(X):
#     """
#     Returns the orthogonal projection matrix of X
#
#     P_ortho = I - X(X^T X)^{-1}X^T
#
#     i.e. Pv projects v onto the orthogonal complement of the columns of X
#     """
#     if len(X.shape) == 1:  # x is a vector
#         return np.eye(len(X)) - np.outer(X, X)
#     else:
#         return np.eye(X.shape[0]) - projection_matrix(X)


def get_svd(X):
    """
    Returns the SVD from numpy
    (does a bit of reformatting on np.linalg.svd)

    X = U D V^T

    Parameters
    ---------
    X: and n x d numpy matrix

    Output
    ------
    (let m = min(n, d))

    U: n x d
    D: list of length m
    V: m x d

    >>> U, D, V = get_numpy_svd(X)
    >>> np.allclose(X, np.dot(U, np.dot(np.diag(D), V.T)))
    True
    """
    U, d, V_T = np.linalg.svd(X, full_matrices=False)
    return U, d, V_T.T


def svd_approx(U, D, V, r):
    """
    Returns the rank r SVD approximation

    X approx U D V.T

    Parameters
    ----------
    (let m = min(n, d))
    U: n x d
    D: list of length m
    V: m x d

    r: rank of desiered approximation
    """
    return np.dot(U[:, 0:r], np.dot(np.diag(D[0:r]), V[:, 0:r].T))


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
    # TODO: what if treu value is zero?
    return np.linalg.norm(true_value - estimate) / np.linalg.norm(true_value)


def absolute_error(true_value, estimate):
    """
    Absolute error under L2/frobenius norm
    """
    return np.linalg.norm(true_value - estimate)
