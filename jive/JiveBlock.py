import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.sparse import issparse

from jive.wedin_bound import get_wedin_samples
from jive.lin_alg_fun import scree_plot

from jive.lazymatpy.templates.matrix_transformations import col_proj, col_proj_orthog
from jive.lin_alg_fun import svd_wrapper, svds_additional

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

        self.init_svd_rank = None

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

        if (self.init_svd_rank is not None) and (signal_rank >= self.init_svd_rank):
            warnings.warn('init_svd_rank must be at least signal_rank + 1 in order to compute singular value threshold to estimate individual rank. You can proceed without this threshold, but will have to manually select teh individual rank.')

            # TODO: decide what the behavior should be here
            # Probably make sv_threshold fail by default, allow user to
            # set it themselves
            self.sv_threshold = np.nan
            # sv_threshold = max(self.sv)
            warnings.warn('no singular value threshold')

        elif not hasattr(self, 'sv'):
            self.sv_threshold = np.nan
            warnings.warn('no singular value threshold')

        else:
            # compute singular value threshold
            self.sv_threshold = get_sv_threshold(self.sv, self.signal_rank)

        # initial signal space estimate
        self.signal_basis = self.scores[:, 0:self.signal_rank]

    def sample_wedin_bound(self, num_samples):

        self.wedin_samples = get_wedin_samples(X=self.X,
                                               U=self.scores,
                                               D=self.sv,
                                               V=self.loadings,
                                               rank=self.signal_rank,
                                               num_samples=num_samples)


    # def compute_wedin_bound(self,
    #                         sampling_procedure=None,
    #                         num_samples=1000,
    #                         quantile='median',
    #                         qr=True):
    #     """
    #     Computes the block wedin bound

    #     Parameters
    #     ----------
    #     sampling_procedure: how to sample vectors from space orthognal to signal subspace
    #     ['svd_resampling' or 'sample_project']. If X is sparse the must use sample_project since
    #     svd_resampling requires the fulll SVD. If None then will use 'svd_resampling' for dense
    #     and 'sample_project' for sparse

    #     num_samples: number of columns to resample for wedin bound

    #     quantile: for wedin bound TODO better description
    #     """
    #     if sampling_procedure is None:
    #         if issparse(self.X):
    #             sampling_procedure ='sample_project'
    #         else:
    #             sampling_procedure ='svd_resampling'

    #     if issparse(self.X) and (sampling_procedure == 'svd_resampling'):
    #         raise ValueError('sparse matrices require sample_project because the full SVD is not computed')

    #     if sampling_procedure == 'svd_resampling':
    #         self.sin_bound_est = get_wedin_bound_svd_basis_resampling(X=self.X,
    #                                                                   U=self.scores,
    #                                                                   D=self.sv,
    #                                                                   V=self.loadings,
    #                                                                   rank=self.signal_rank,
    #                                                                   num_samples=num_samples,
    #                                                                   quantile=quantile)

    #     elif sampling_procedure == 'sample_project':
    #         self.sin_bound_est = get_wedin_bound_sample_project(X=self.X,
    #                                                             U=self.scores,
    #                                                             D=self.sv,
    #                                                             V=self.loadings,
    #                                                             rank=self.signal_rank,
    #                                                             num_samples=num_samples,
    #                                                             quantile=quantile,
    #                                                             qr=qr)
    #     else:
    #         raise ValueError('sampling_procedure must be one of svd_resampling or sample_project')

    #     # TODO: maybe give user option to kill these
    #     # if kill_init_svd:
    #     #     self.scores = None
    #     #     self.loadings = None
    #     #    self.sv = None

    def compute_final_decomposition(self,
                                    joint_scores,
                                    individual_rank=None,
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
                                                                       signal_rank=self.signal_rank,
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
                              signal_rank=None, save_full_estimate=False):
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

        # largest the individual rank can be
        # TODO: check this
        joint_rank = joint_scores.shape[1]
        max_rank = min(X.shape) - joint_rank

        if issparse(X):

            # compute a rank R1 SVD of I
            # if the estimated individual rank is less than R1 we are done
            # otherwise compute a rank R2 SVD of I
            # keep going until we find the individual rank

            # TODO: this could use lots of optimizing
            current_rank = min(int(1.5 * signal_rank), max_rank) # 1.2 is somewhat arbitrary
            scores, sv, loadings = svd_wrapper(I, current_rank)
            individual_rank = sum(sv > sv_threshold)

            if individual_rank == current_rank:

                found_indiv_rank = False
                for t in range(3):

                    # current guess at an upper bound for the individual rank
                    additional_rank = signal_rank

                    current_rank = current_rank + additional_rank
                    current_rank = min(current_rank, max_rank)

                    # compute additional additional_rank SVD components

                    # TODO: possibly use svds_additional to speed up calculation
                    # scores, sv, loadings = svds_additional(I, scores, sv, loadings, additional_rank)
                    scores, sv, loadings = svd_wrapper(I, current_rank)
                    individual_rank = sum(sv > sv_threshold)

                    # we are done if the individual rank estimate is less
                    # than the current_rank or if the current_rank is equal to the maximal rank
                    if (individual_rank < current_rank) or (current_rank == max_rank):
                        found_indiv_rank = True
                        break

                if not found_indiv_rank:
                    warnings.warn('individual rank estimate probably too low')


        else:
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

    return I, scores, sv, loadings


