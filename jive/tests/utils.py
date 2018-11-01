import numpy as np


def svd_checker(U, D, V, n, d, rank):
    passes = True

    if U.shape != (n, rank):
        passes = False

    if D.shape != (rank, ):
        passes = False

    if V.shape != (d, rank):
        passes = False

    if not np.allclose(np.dot(U.T, U), np.eye(rank)):
        passes = False

    if not np.allclose(np.dot(V.T, V), np.eye(rank)):
        passes = False

    # check singular values are non-increasing
    for i in range(len(D) - 1):
        if D[i] > D[i+1]:
            passes = False

    return passes

