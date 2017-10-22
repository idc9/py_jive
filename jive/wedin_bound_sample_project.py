import numpy as np
from sklearn.utils.extmath import safe_sparse_dot

def get_wedin_bound_sample_project(X, U, D, V, rank, num_samples=1000, quantile='median'):
    """
    Computes the wedin bound using the sample-project procedure. This method
    does not require the full SVD.

    Parameters
    ----------
    X: the data block
    U, D, V: the partial SVD of X
    rank: the rank of the signal space
    num_samples: number of saples for resampling procedure
    quantile: TODO desc, implement this
    """

    # TODO: implement this
    if quantile != 'median':
        raise NotImplemented

    # resample for U and V
    U_norm_samples = norms_sample_project(X=X.T,
                                          B=U[:, 0:rank],
                                          rank=rank,
                                          num_samples=num_samples)

    V_norm_samples = norms_sample_project(X=X,
                                          B=V[:, 0:rank],
                                          rank=rank,
                                          num_samples=num_samples)

    # compute upper bound
    # TODO: which way?
    # EV_estimate = np.median(V_norm_samples)
    # UE_estimate = np.median(U_norm_samples)
    # wedin_bound_est = max(EV_estimate, UE_estimate)/sigma_min
    sigma_min = D[rank - 1]  # TODO: double check -1
    wedin_bound_samples = [max(U_norm_samples[s], V_norm_samples[s])/sigma_min for s in range(num_samples)]
    wedin_bound_est = np.median(wedin_bound_samples)

    return wedin_bound_est

def norms_sample_project(X, B, rank, num_samples=1000):
    """
    Samples vectors from space orthognal to signal space as follows
    - sample random vector from isotropic distribution
    - project onto orthogonal complement of signal space and normalize


    Parameters
    ---------
    X: the observed data
    
    B: the basis for the signal col/rows space (e.g. the left/right singular vectors)

    rank: number of columns to resample

    num_samples: how many resamples

    Output
    ------
    an array of the resampled norms
    """
    dim = B.shape[0]

    sampled_norms = [0]*num_samples


    for s in range(num_samples):

        # sample from isotropic distribution
        # TODO: maybe uniform?
        vecs = np.random.normal(size=(dim, rank))

        # project onto space orthogonal to cols of B
        vecs = (np.eye(dim) - np.dot(B, B.T)).dot(vecs)

        # normalize vectors
        vecs = np.vstack([unit_vector(vecs[:, c]) for c in range(vecs.shape[1])]).T

        # TODO: maybe give option to create orthogonal basis using QR decomp

        # compute  operator L2 norm
        sampled_norms[s] = np.linalg.norm(X.dot(vecs), ord=2)

    return sampled_norms



def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    vector = np.asarray(vector)
    ret =  vector / np.linalg.norm(vector)
    return ret.reshape(-1)