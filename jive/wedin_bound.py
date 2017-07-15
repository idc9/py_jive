import numpy as np


def get_wedin_bound(X, U, D, V, rank, num_samples=1000):
    """
    Computes the wedin bound using the resampling procedure described in
    the AJIVE paper.

    Parameters
    ----------
    X: the data block
    U, D, V: the SVD of X
    rank: the rank of the signal space
    num_samples: number of saples for resampling procedure
    """

    # resample for U and V
    U_sampled_norms = resampled_wedin_bound(X=X,
                                            orthogonal_basis=U[:, rank:],
                                            rank=rank,
                                            right_vectors=False,
                                            num_samples=num_samples)

    V_sampled_norms = resampled_wedin_bound(X=X,
                                            orthogonal_basis=V[:, rank:],
                                            rank=rank,
                                            right_vectors=True,
                                            num_samples=num_samples)

    # compute upper bound
    # TODO: which way?
    # EV_estimate = np.median(V_sampled_norms)
    # UE_estimate = np.median(U_sampled_norms)
    # wedin_bound_est = max(EV_estimate, UE_estimate)/sigma_min
    sigma_min = D[rank - 1]  # TODO: double check -1
    wedin_bound_samples = [max(U_sampled_norms[s], V_sampled_norms[s])/sigma_min for s in range(num_samples)]
    wedin_bound_est = np.median(wedin_bound_samples)

    return wedin_bound_est


def resampled_wedin_bound(X, orthogonal_basis, rank,
                          right_vectors, num_samples=1000):
    """
    Resampling procedure described in AJIVE paper for Wedin bound

    Parameters
    ---------
    orthogonal_basis: basis vectors for the orthogonal complement
    of the score space

    X: the observed data

    rank: number of columns to resample

    right_vectors: multiply right or left of data matrix (True/False)

    num_samples: how many resamples

    Output
    ------
    an array of the resampled norms
    """

    resampled_norms = [0]*num_samples

    for s in range(num_samples):

        # sample columns from orthogonal basis
        sampled_col_index = np.random.choice(orthogonal_basis.shape[1], size=rank, replace=True)
        resampled_basis = orthogonal_basis[:, sampled_col_index]
        # ^ this is V* from AJIVE p12

        # project observed data
        if right_vectors:
            resampled_projection = np.dot(X, resampled_basis)
        else:
            resampled_projection = np.dot(X.T, resampled_basis)

        # compute resampled operator L2 nrm
        resampled_norms[s] = np.linalg.norm(resampled_projection, ord=2)

    return rs_norms
