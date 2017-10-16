import numpy as np
import matplotlib.pyplot as plt

from jive.wedin_bound import get_wedin_bound
from jive.lin_alg_fun import get_svd, scree_plot

from scipy.sparse import issparse
from scipy.sparse.linalg import svds
from scipy.linalg import svd as full_svd

from py_fun_iain.sparse.GenSpMatPPtS import GenSpMatPPtS
from py_fun_iain.sparse.GenSpMatI_PPtS import GenSpMatI_PPtS
from py_fun_iain.sparse.GenSparseMat import is_gen_sparse_matrix

class JiveBlock(object):

    def __init__(self, X, init_svd_rank, save_full_final_decomp, name=None):
        """
        Stores a single data block (rows as observations).

        Paramters
        ---------
        X: a data block

        init_svd_rank: rank of the first SVD before estimating the signal
        rank. Optional unless the matrix is sparse.

        save_full_final_decomp: whether or not to save the full I, J, E matrices

        name: the (optional) name of the block
        """
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]

        self.name = name

        # Options
        self.save_full_final_decomp = save_full_final_decomp

        # Compute initial SVD
        if issparse(X) and (init_svd_rank is None):
            raise ValueError('sparse matrices must have an init_svd_rank')

        if (init_svd_rank is not None) and ((init_svd_rank < 1) or (init_svd_rank > min(self.n, self.d))):
            raise ValueError('init_svd_rank must be between 1 and min(n, d)')

        self.init_svd_rank = init_svd_rank
        self.initial_svd()

    def initial_svd(self):
        """
        SVD for initial signal space extraction
        """
        # TODO: rename these to scores, sv and loadings
        self.U, self.D, self.V = svd_wrapper(self.X, self.init_svd_rank)

    def scree_plot(self):
        """
        Plots scree plot for initial SVD
        """
        if self.name:
            title = self.name
        else:
            title = ''
        scree_plot(self.D, 'X scree plot', diff=False, title=title)

    def set_signal_rank(self, signal_rank):
        """
        The user sets the signal ranks for the data block. Then compute
        - the singular value threshold
        - the signal space basis
        - the wedin bound
        """
        self.signal_rank = signal_rank

        # compute singular value threshold
        self.sv_threshold = get_sv_threshold(self.D, self.signal_rank)

        # initial signal space estimate
        self.signal_basis = self.U[:, 0:self.signal_rank]

        # TODO: maybe give user the option to kill U, D, V at this point

    def compute_wedin_bound(self):
        # use resampling to compute wedin bound estimate
        self.wedin_bound = get_wedin_bound(X=self.X,
                                           U=self.U,
                                           D=self.D,
                                           V=self.V,
                                           rank=self.signal_rank,
                                           num_samples=1000)

        # I think I can kill these now to save memory
        self.U = None
        self.V = None
        self.D = None

    def final_decomposition(self, joint_scores):
        """
        Compute the JIVE decomposition of the block.

        Parameters
        ---------
        joint_scores: joint scors matrix
        """
        self.estimate_joint_space(joint_scores)

        self.estimate_individual_space(joint_scores)

        # save ranks
        self.individual_rank = len(self.block_individual_sv)
        self.joint_rank = joint_scores.shape[1]  # not necessary to save

        # estimate noise matrix
        if self.save_full_final_decomp:
            self.E = self.X - (self.J + self.I)
            # self.X = None  # TODO: I think I can kill this here
        else:
            self.E = np.matrix([])

    def estimate_joint_space(self, joint_scores):
        """
        Estimate the block's joint space
        """
        self.J, self.block_joint_scores,  self.block_joint_sv, \
        self.block_joint_loadings = get_block_joint_space(X=self.X,
                                                          joint_scores=joint_scores,
                                                          save_full_final_decomp=self.save_full_final_decomp)

    def estimate_individual_space(self, joint_scores):
        """
        Estimate the block's individual space
        """
        self.I, self.block_individual_scores, \
            self.block_individual_sv, self.block_individual_loadings = \
            get_block_individual_space(X=self.X,
                                       joint_scores=joint_scores,
                                       sv_threshold=self.sv_threshold,
                                       save_full_final_decomp=self.save_full_final_decomp)

    def get_jive_estimates(self):
        """
        Returns all all estimated JIVE information including J, I, E and the
        respective SVDs
        """
        estimates = {}
        estimates['joint'] = {'full': self.J,
                              'U': self.block_joint_scores,
                              'D': self.block_joint_sv,
                              'V': self.block_joint_loadings,
                              'rank': self.joint_rank}

        estimates['individual'] = {'full': self.I,
                                   'U': self.block_individual_scores,
                                   'D': self.block_individual_sv,
                                   'V': self.block_individual_loadings,
                                   'rank': self.individual_rank}

        estimates['noise'] = self.E

        return estimates

    def get_full_jive_estimates(self):
        """
        Returns only the full JIVE estimaes J, I, E
        """
        estimates = {}
        if self.save_full_final_decomp:
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

            # TODO: I think I can kill this here
            # self.X = None

        return estimates


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


def get_block_joint_space(X, joint_scores, save_full_final_decomp=True):
    """"
    Finds a block's joint space representation and the SVD of this space.

    Paramters
    ---------
    X: observed data block

    joint_scores: joint scores matrix for joint space

    save_full_final_decomp: whether or not to return the full J matrix

    Output
    ------
    J, block_joint_scores, block_joint_sv, block_joint_loadings

    note the last three terms are the SVD approximation of I
    """

    # # compute full block joint represntation
    # joint_projection = np.dot(joint_scores, joint_scores.T)
    # J = np.dot(joint_projection, X)

    # # compute block joint SVD
    # block_joint_scores, block_joint_sv, block_joint_loadings = get_svd(J)

    if issparse(X):
        J = GenSpMatPPtS(X, joint_scores)

    else:
        J = np.dot(joint_scores, np.dot(joint_scores.T, X))


    joint_rank = joint_scores.shape[1]
    scores, sv, loadings = svd_wrapper(J, joint_rank)

    if not save_full_final_decomp:
        J = np.matrix([])  # kill J matrix to save memory

    return J, scores, sv, loadings


def get_block_individual_space(X, joint_scores, sv_threshold, save_full_final_decomp=True):
    """"
    Finds a block's individual space representation and the SVD of this space

    Paramters
    ---------
    X: observed data block

    joint_scores: joint scores matrix for joint space

    sv_threshold: singular value threshold

    save_full_final_decomp: whether or not to return the full I matrix

    Output
    ------
    I, block_individual_scores, block_individual_sv, block_individual_loadings

    note the last three terms are the SVD approximation of I
    """

    # joint_projection_othogonal = np.eye(X.shape[0]) - \
    #     np.dot(joint_scores, joint_scores.T)

    # # SVD of orthogonal projection
    # X_ortho = np.dot(joint_projection_othogonal, X)
    # block_individual_scores, block_individual_sv, block_individual_loadings = get_svd(X_ortho)

    if issparse(X):
        I = GenSpMatI_PPtS(X, joint_scores)

    else:
        I = X - np.dot(joint_scores, np.dot(joint_scores.T, X))


    joint_rank = joint_scores.shape[1]
    scores, sv, loadings = svd_wrapper(I, joint_rank)

    # compute individual rank
    individual_rank = sum(sv > sv_threshold)
    scores = scores[:, 0:individual_rank]
    sv = sv[0:individual_rank]
    loadings = loadings[:, 0:individual_rank]

    # full block individual representation
    if save_full_final_decomp:
        I = np.dot(scores, np.dot(np.diag(sv), loadings.T))
    else:
        I = np.matrix([])  # Kill I matrix to save memory

    return I, scores, sv, loadings


def svd_wrapper(X, rank = None):
    """
    Computes the (possibly partial) SVD of a matrix. Handles the case where
    X is either dense or sparse.
    
    Parameters
    ----------
    X: either dense or sparse
    rank: rank of the desired SVD (required for sparse matrices)

    Output
    ------
    U, D, V
    the columns of U are the left singular vectors
    the COLUMNS of V are the left singular vectors

    """
    if is_gen_sparse_matrix(X):
        scipy_svds = svds(X.as_linear_operator(), rank)
        U, D, V = fix_scipy_svds(scipy_svds)
        V = V.T
    
    elif issparse(X):
        scipy_svds = svds(X, rank)
        U, D, V = fix_scipy_svds(scipy_svds)
        V = V.T
        
    else:
        U, D, V = full_svd(X, full_matrices=False)
        V = V.T

    if rank:
        U = U[:, :rank]
        D = D[:rank]
        V = V[:, :rank]
        
    return U, D, V


def fix_scipy_svds(scipy_svds):
    """
    scipy.sparse.linalg.svds orders the singular values backwards,
    this function fixes this insanity and returns the singular values
    in decreasing order
    
    Parameters
    ----------
    scipy_svds: the out put from scipy.sparse.linalg.svds
    
    Output
    ------
    U, D, V
    ordered in decreasing singular values
    """
    U, D, V = scipy_svds

    U = U[:, ::-1]
    D = D[::-1]
    V = V[::-1, :]

    return U, D, V
