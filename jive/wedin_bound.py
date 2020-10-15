import numpy as np
from joblib import Parallel, delayed


def get_wedin_samples(X, U, D, V, rank, R=1000, n_jobs=None):
    """
    Computes the wedin bound using the sample-project procedure. This method
    does not require the full SVD.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        The data block.

    U, D, V:
        The partial SVD of X

    rank: int
        The rank of the signal space

    R: int
        Number of samples for resampling procedure

    n_jobs: int, None
        Number of jobs for parallel processing using
        sklearn.externals.joblib.Parallel. If None, will not use parallel
        processing.

    device: str, None
        If not None, will do computations on GPU using pytorch. device is the
        name of the gpu device to use. Either device or n_jobs can be used,
        but not both.

    """

    if n_jobs is not None:
        basis = V[:, 0:rank]
        V_norm_samples = Parallel(n_jobs=n_jobs)\
            (delayed(_get_sample)(X, basis) for i in range(R))

        basis = U[:, 0:rank]
        U_norm_samples = Parallel(n_jobs=n_jobs)\
            (delayed(_get_sample)(X.T, basis) for i in range(R))

    else:
        basis = V[:, 0:rank]
        V_norm_samples = [_get_sample(X, basis) for r in range(R)]

        basis = U[:, 0:rank]
        U_norm_samples = [_get_sample(X.T, basis) for r in range(R)]

    V_norm_samples = np.array(V_norm_samples)
    U_norm_samples = np.array(U_norm_samples)

    sigma_min = D[rank - 1]  # TODO: double check -1
    wedin_bound_samples = [min(max(U_norm_samples[r],
                                   V_norm_samples[r]) / sigma_min, 1)
                           for r in range(R)]

    return wedin_bound_samples


def _get_sample(X, basis):
    dim, rank = basis.shape

    # sample from isotropic distribution
    vecs = np.random.normal(size=(dim, rank))

    # project onto space orthogonal to cols of B
    # vecs = (np.eye(dim) - np.dot(basis, basis.T)).dot(vecs)
    vecs = vecs - np.dot(basis, np.dot(basis.T, vecs))

    # orthonormalize
    vecs, _ = np.linalg.qr(vecs)

    # compute  operator L2 norm
    return np.linalg.norm(X.dot(vecs), ord=2)
