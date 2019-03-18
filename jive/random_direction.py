import numpy as np
from jive.utils import svd_wrapper
from sklearn.externals.joblib import Parallel, delayed
import torch


def sample_randdir(num_obs, signal_ranks, R=1000, n_jobs=None, device=None):
    """
    Draws samples for the random direction bound.

    Parameters
    ----------

    num_obs: int
        Number of observations.

    signal_ranks: list of ints
        The initial signal ranks for each block.

    R: int
        Number of samples to draw.

    n_jobs: int, None
        Number of jobs for parallel processing using
        sklearn.externals.joblib.Parallel. If None, will not use parallel
        processing.

    device: str, None
        If not None, will do computations on GPU using pytorch. device is the
        name of the gpu device to use. Either device or n_jobs can be used,
        but not both.

    Output
    ------
    random_sv_samples: np.array, shape (R, )
        The samples.
    """

    if sum((n_jobs is not None, device is not None)) <= 1:
        raise ValueError('At most one of n_jobs and device can be not None.')

    if n_jobs is not None:
        random_sv_samples = Parallel(n_jobs=n_jobs)\
            (delayed(_get_rand_sample)(num_obs, signal_ranks)
             for i in range(R))

    elif device is not None:  # use pytorch backend
        random_sv_samples = [_get_rand_sample_pytorch(num_obs, signal_ranks,
                                                      device=device)
                             for r in range(R)]

    else:
        random_sv_samples = [_get_rand_sample(num_obs, signal_ranks)
                             for r in range(R)]

    return np.array(random_sv_samples)


def _get_rand_sample(num_obs, signal_ranks):
    M = [None for _ in range(len(signal_ranks))]
    for k in range(len(signal_ranks)):

        # sample random orthonormal basis
        Z = np.random.normal(size=(num_obs, signal_ranks[k]))
        M[k] = np.linalg.qr(Z)[0]

    # compute largest sing val of random joint matrix
    M = np.bmat(M)
    _, svs, __ = svd_wrapper(M, rank=1)

    return svs.item() ** 2


def _get_rand_sample_pytorch(num_obs, signal_ranks, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    M = [None for _ in range(len(signal_ranks))]

    with torch.no_grad():
        for k in range(len(signal_ranks)):

            # sample random orthonormal basis
            Z = torch.randn(size=(num_obs, signal_ranks[k]), device=device)
            M[k] = torch.qr(Z)[0]

        M = torch.cat(M, dim=1)
        M = M.cpu().detach().numpy()

    # compute largest sing val of random joint matrix
    _, svs, __ = svd_wrapper(M, rank=1)

    return svs.item() ** 2
