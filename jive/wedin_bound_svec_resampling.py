import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

def get_wedin_bound_svec_resampling(X, U, D, V, rank, num_samples=1000, quantile='median'):
    """
    Computes the wedin bound using the resampling procedure described in
    the AJIVE paper. This procedure requires the full SVD of X.

    Parameters
    ----------
    X: the data block
    U, D, V: the SVD of X
    rank: the rank of the signal space
    num_samples: number of saples for resampling procedure
    quantile: TODO desc, implement this
    """

    # TODO: implement this
    if quantile != 'median':
        raise NotImplemented

    # resample for U and V
    U_sampled_norms = norms_svec_resampling(X=X.T,
                                            orthogonal_basis=U[:, rank:],
                                            rank=rank,
                                            num_samples=num_samples)

    V_sampled_norms = norms_svec_resampling(X=X,
                                            orthogonal_basis=V[:, rank:],
                                            rank=rank,
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


def norms_svec_resampling(X, orthogonal_basis, rank, num_samples=1000):
    """
    Resampling procedure described in AJIVE paper for Wedin bound

    Parameters
    ---------
    orthogonal_basis: basis vectors for the orthogonal complement
    of the score space

    X: the observed data

    rank: number of columns to resample

    num_samples: how many resamples

    Output
    ------
    an array of the resampled norms
    """
    # TODO: rename some arguments e.g. orthogonal_basis -- this is not really
    # a basis, also resampled_projection

    resampled_norms = [0]*num_samples

    for s in range(num_samples):

        # sample columns from orthogonal basis
        sampled_col_index = np.random.choice(orthogonal_basis.shape[1], size=rank, replace=True)
        resampled_basis = orthogonal_basis[:, sampled_col_index]
        # ^ this is V* from AJIVE p12

        # project observed data

        # resampled_projection = np.dot(X, resampled_basis)
        resampled_projection = safe_sparse_dot(X, resampled_basis)


        # compute resampled operator L2 norm
        resampled_norms[s] = np.linalg.norm(resampled_projection, ord=2)

    return resampled_norms