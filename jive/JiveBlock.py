import numpy as np
import matplotlib.pyplot as plt

from jive.wedin_bound_sample_project import get_wedin_bound_sample_project
from jive.wedin_bound_svec_resampling import get_wedin_bound_svec_resampling
from jive.lin_alg_fun import scree_plot

from scipy.sparse import issparse

from jive.lazymatpy.templates.matrix_transformations import col_proj, col_proj_orthog
from jive.lin_alg_fun import svd_wrapper

class JiveBlock(object):

    def __init__(self, X, name=None):
        """
        Stores a single data block (rows as observations).

        Paramters
        ---------
        X: a data block

        name: (optional) the name of the block
        """
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]

        self.name = name

    def initial_svd(self, init_svd_rank):
        """
        SVD for initial signal space extraction
        """   
        if issparse(self.X) and (init_svd_rank is None):
            raise ValueError('sparse matrices must have an init_svd_rank')

        if (init_svd_rank is not None) and ((init_svd_rank < 1) or (init_svd_rank > min(self.n, self.d) - 1)):
            raise ValueError('init_svd_rank must be between 1 and min(n, d)')

        self.init_svd_rank = init_svd_rank

        self.scores, self.sv, self.loadings = svd_wrapper(self.X, self.init_svd_rank)

    def scree_plot(self, log, diff):
        """
        Plots scree plot for initial SVD
        """
        if not hasattr(self, 'sv'):
            raise ValueError('please run initial_svd() before making scree plot')


        if self.name:
            title = self.name
        else:
            title = ''
        scree_plot(self.sv, log=log, diff=diff,  title=title)

    def set_signal_rank(self, signal_rank):
        """
        User set signal rank
        """
        self.signal_rank = signal_rank

        # compute singular value threshold
        self.sv_threshold = get_sv_threshold(self.sv, self.signal_rank)

        # initial signal space estimate
        self.signal_basis = self.scores[:, 0:self.signal_rank]


    def compute_wedin_bound(self, sampling_procedure=None, num_samples=1000, quantile='median'):
        """
        Computes the block wedin bound

        Parameters
        ----------
        sampling_procedure: how to sample vectors from space orthognal to signal subspace
        ['svec_resampling' or 'sample_project']. If X is sparse the must use sample_project since
        svec_resampling requires the fulll SVD. If None then will use 'svec_resampling' for dense
        and 'sample_project' for sparse

        num_samples: number of columns to resample for wedin bound

        quantile: for wedin bound TODO better description
        """
        if sampling_procedure is None:
            if issparse(self.X):
                sampling_procedure ='sample_project'
            else:
                sampling_procedure ='svec_resampling'

        if issparse(self.X) and (sampling_procedure == 'svec_resampling'):
            raise ValueError('sparse matrices require sample_project because the full SVD is not computed')

        if sampling_procedure == 'svec_resampling':
            self.sin_bound_est = get_wedin_bound_svec_resampling(X=self.X,
                                                                 U=self.scores,
                                                                 D=self.sv,
                                                                 V=self.loadings,
                                                                 rank=self.signal_rank,
                                                                 num_samples=num_samples,
                                                                 quantile=quantile)

        elif sampling_procedure == 'sample_project':
            self.sin_bound_est = get_wedin_bound_sample_project(X=self.X,
                                                                U=self.scores,
                                                                D=self.sv,
                                                                V=self.loadings,
                                                                rank=self.signal_rank,
                                                                num_samples=num_samples,
                                                                quantile=quantile)
        else:
            raise ValueError('sampling_procedure must be one of svec_resampling or sample_project')

        # TODO: maybe give user option to kill these
        # if kill_init_svd:
        #     self.scores = None
        #     self.loadings = None
        #    self.sv = None

    def compute_final_decomposition(self, joint_scores, individual_rank=None,
                                    save_full_estimate=False):
        """
        Compute the JIVE decomposition of the block.

        Parameters
        ---------
        joint_scores: joint scors matrix

        individual_rank: give user the option ot specify the individual rank

        save_full_estimate: whether or not to save the full I, J, E matrices
        """

        self.save_full_estimate = save_full_estimate

        self.estimate_joint_space(joint_scores)
        self.estimate_individual_space(joint_scores, individual_rank)

        # save ranks
        self.individual_rank = self.block_individual_scores.shape[1]
        self.joint_rank = joint_scores.shape[1]  # not necessary to save

        # estimate noise matrix
        if self.save_full_estimate:
            self.E = self.X - (self.J + self.I)
        else:
            self.E = np.array([])

    def estimate_joint_space(self, joint_scores):
        """
        Estimate the block's joint space
        """
        if joint_scores.shape[1] == 0:
            self.J = np.array([])
            self.block_joint_scores = np.array([])
            self.block_joint_sv = np.array([])
            self.block_joint_loadings = np.array([])

        else:
            self.J, self.block_joint_scores, self.block_joint_sv, \
            self.block_joint_loadings = estimate_joint_space(X=self.X,
                                                             joint_scores=joint_scores,
                                                             save_full_estimate=self.save_full_estimate)

    def estimate_individual_space(self, joint_scores, individual_rank=None):
        """
        Estimate the block's individual space
        """

        self.I, self.block_individual_scores, self.block_individual_sv, \
            self.block_individual_loadings = estimate_individual_space(X=self.X,
                                                                       joint_scores=joint_scores,
                                                                       sv_threshold=self.sv_threshold,
                                                                       individual_rank=individual_rank,
                                                                       init_svd_rank=self.init_svd_rank,
                                                                       save_full_estimate=self.save_full_estimate)

    def get_block_estimates(self):
        """
        Returns all all estimated JIVE information including J, I, E and the
        respective SVDs
        """
        estimates = {}
        estimates['joint'] = {'full': self.J,
                              'scores': self.block_joint_scores,
                              'sing_vals': self.block_joint_sv,
                              'loadings': self.block_joint_loadings,
                              'rank': self.joint_rank}

        estimates['individual'] = {'full': self.I,
                                   'scores': self.block_individual_scores,
                                   'sing_vals': self.block_individual_sv,
                                   'loadings': self.block_individual_loadings,
                                   'rank': self.individual_rank}

        estimates['noise'] = self.E

        return estimates

    def get_full_estimates(self):
        """
        Returns only the full JIVE estimaes J, I, E
        """
        estimates = {}
        if self.save_full_estimate:
            estimates['J'] = self.J
            estimates['I'] = self.I
            estimates['E'] = self.E
        else:
            # compute full matrices on the spot.
            estimates['J'] = np.dot(self.block_joint_scores,
                                    np.dot(np.diag(self.block_joint_sv),
                                           self.block_joint_loadings.T))

            estimates['I'] = np.dot(self.block_individual_scores,
                                    np.dot(np.diag(self.block_individual_sv),
                                           self.block_individual_loadings.T))

            estimates['E'] = self.X - (estimates['I'] + estimates['J'])

        return estimates

    def kill_init_svd(self):
        """
        Kills initial svd scores/loadings to save memory
        """
        # TODO: is this worth adding?

        # self.score = None
        # self.loadings = None
        raise NotImplementedError


def get_sv_threshold(singular_values, rank):
    """
    Returns the singular value threshold value for rank R; half way between
    the thresholding value and the next smallest.

    Paramters
    ---------
    singular_values: list of singular values

    rank: rank of the threshold
    """
    # Note the zero indexing so rank = the rank + 1th singular value
    return .5 * (singular_values[rank - 1] + singular_values[rank])


def estimate_joint_space(X, joint_scores, save_full_estimate=False):
    """"
    Finds a block's joint space representation and the SVD of this space.

    Paramters
    ---------
    X: observed data block

    joint_scores: joint scores matrix for joint space

    save_full_estimate: whether or not to return the full J matrix

    Output
    ------
    J, block_joint_scores, block_joint_sv, block_joint_loadings

    note the last three terms are the SVD approximation of I
    """

    if issparse(X): # lazy evaluation for sparse matrices
        J = col_proj(X, joint_scores)
    else:
        J = np.dot(joint_scores, np.dot(joint_scores.T, X))


    joint_rank = joint_scores.shape[1]
    scores, sv, loadings = svd_wrapper(J, joint_rank)

    if not save_full_estimate:
        J = np.array([])  # kill J matrix to save memory

    return J, scores, sv, loadings


def estimate_individual_space(X, joint_scores, sv_threshold, individual_rank=None,
                              init_svd_rank=None, save_full_estimate=False):
    """"
    Finds a block's individual space representation and the SVD of this space

    Paramters
    ---------
    X: observed data block

    joint_scores: joint scores matrix for joint space

    sv_threshold: singular value threshold

    individual_rank: maunally set individual rank

    save_full_estimate: whether or not to return the full I matrix

    Output
    ------
    I, block_individual_scores, block_individual_sv, block_individual_loadings

    note the last three terms are the SVD approximation of I
    """

    # project columns of X onto orthogonal complement to joint_scores
    if joint_scores.shape[1] == 0:
        I = X
    elif issparse(X):  # lazy evaluation for sparse matrices
        I = col_proj_orthog(X, joint_scores)
    else:
        I = X - np.dot(joint_scores, np.dot(joint_scores.T, X))

    # estimate individual rank
    if individual_rank is None:
        # SVD of projected matrix
        # TODO: better bound on rank
        # TODO: maybe make this incremental i.e. compute rank R svd,
        # if the estimated indiv rank is is equal to R then compute 
        # R + 1 svd, etc 
        max_rank = min(X.shape) - joint_scores.shape[1]
        if init_svd_rank is not None:
            max_rank = min(init_svd_rank, max_rank)

        scores, sv, loadings = svd_wrapper(I, max_rank)

        # compute individual rank
        individual_rank = sum(sv > sv_threshold)
        
        scores = scores[:, 0:individual_rank]
        sv = sv[0:individual_rank]
        loadings = loadings[:, 0:individual_rank]
    
    else:
        scores, sv, loadings = svd_wrapper(I, individual_rank)

    # full block individual representation
    if save_full_estimate:
        I = np.dot(scores, np.dot(np.diag(sv), loadings.T))
    else:
        I = np.array([])  # Kill I matrix to save memory

    print(np.allclose(np.dot(loadings.T, loadings), np.eye(loadings.shape[1])))
    return I, scores, sv, loadings


