import numpy as np
from jive.utils import svd_wrapper


def sample_randdir(num_obs, signal_ranks, num_samples=1000):
    """
    TODO: document
    TODO: parallelize
    """
    random_sv_samples = np.zeros(num_samples)
    num_blocks = len(signal_ranks)
    for i in range(num_samples):
        M = [None for _ in range(num_blocks)]
        for k in range(num_blocks):

            # sample random orthonormal basis
            Z = np.random.normal(size=(num_obs, signal_ranks[k]))
            M[k] = np.linalg.qr(Z)[0]

        # compute largest sing val of random joint matrix
        M = np.bmat(M)
        _, svs, __ = svd_wrapper(M)
        random_sv_samples[i] = max(svs) ** 2

    return random_sv_samples
