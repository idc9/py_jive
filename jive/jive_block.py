import numpy as np
import matplotlib.pyplot as plt

from wedin_bound import get_wedin_bound
from lin_alg_fun import get_svd, svd_approx, scree_plot


class JiveBlock(object):

    def __init__(self, X, full=True, name=None):
        """
        Stores a single data block (rows as observations).

        Paramters
        ---------
        X: a data block

        full: whether or not to save the full I, J, E matrices

        name: the (optional) name of the block
        """
        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]

        self.name = name

        # Options
        self.full = full

        # Compute initial SVD
        self.initial_svd()

    def initial_svd(self):
        """
        SVD for initial signal space extraction
        """
        self.U, self.D, self.V = get_svd(X)

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
        if self.full:
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
                                                          full=self.full)

    def estimate_individual_space(self, joint_scores):
        """
        Estimate the block's individual space
        """
        self.I, self.block_individual_scores, \
            self.block_individual_sv, self.block_individual_loadings = \
            get_block_individual_space(X=self.X,
                                       joint_scores=joint_scores,
                                       sv_threshold=self.sv_threshold,
                                       full=self.full)

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
        if self.full:
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


def get_block_joint_space(X, joint_scores, full=True):
    """"
    Finds a block's joint space representation and the SVD of this space.

    Paramters
    ---------
    X: observed data block

    joint_scores: joint scores matrix for joint space

    full: whether or not to return the full J matrix

    Output
    ------
    J, block_joint_scores, block_joint_sv, block_joint_loadings

    note the last three terms are the SVD approximation of I
    """

    # compute full block joint represntation
    joint_projection = np.dot(joint_scores, joint_scores.T)
    J = np.dot(joint_projection, X)

    # compute block joint SVD
    block_joint_scores, block_joint_sv, block_joint_loadings = get_svd(J)
    joint_rank = joint_scores.shape[1]
    block_joint_scores = block_joint_scores[:, 0:joint_rank]
    block_joint_sv = block_joint_sv[0:joint_rank]
    block_joint_loadings = block_joint_loadings[:, 0:joint_rank]

    if not full:
        J = np.matrix([])  # kill J matrix to save memory

    return J, block_joint_scores, block_joint_sv, block_joint_loadings


def get_block_individual_space(X, joint_scores, sv_threshold, full=True):
    """"
    Finds a block's individual space representation and the SVD of this space

    Paramters
    ---------
    X: observed data block

    joint_scores: joint scores matrix for joint space

    sv_threshold: singular value threshold

    full: whether or not to return the full I matrix

    Output
    ------
    I, block_individual_scores, block_individual_sv, block_individual_loadings

    note the last three terms are the SVD approximation of I
    """

    joint_projection_othogonal = np.eye(X.shape[0]) - \
        np.dot(joint_scores, joint_scores.T)

    # SVD of orthogonal projection
    X_ortho = np.dot(joint_projection_othogonal, X)
    block_individual_scores, block_individual_sv, block_individual_loadings = get_svd(X_ortho)

    # compute individual rank
    individual_rank = sum(block_individual_sv > sv_threshold)
    block_individual_scores = block_individual_scores[:, 0:individual_rank]
    block_individual_sv = block_individual_sv[0:individual_rank]
    block_individual_loadings = block_individual_loadings[:, 0:individual_rank]

    # full block individual representation
    if full:
        I = np.dot(block_individual_scores,
                   np.dot(np.diag(block_individual_sv),
                          block_individual_loadings.T))
    else:
        I = np.matrix([])  # Kill I matrix to save memory

    return I, block_individual_scores, block_individual_sv, block_individual_loadings


def block_JIVE_decomposition(X, joint_scores, sv_threshold):
    """
    CURRENTLY NOT USED
    Computes the JIVE decomposition for an individual data block. Only returns
    J, I, E and individual rank.

    Parameters
    ----------
    X: observed data matrix

    joint_scores: joint scores

    sv_threshold: singular value threshold

    Output
    ------
    J, I, E, individual_rank
    J: estimated joint signal matrix
    I: estimated indiviual signal
    E: esimated error matrix
    individual_rank: rank of the esimated individual space
    """
    # TODO: maybe reorder projection computation
    # compute orthogonal projection matrices
    joint_projection = np.dot(joint_scores, joint_scores.T)
    joint_projection_ortho = np.eye(X.shape[0]) - joint_projection

    # estimate joint spaces
    J = np.dot(joint_projection, X)

    # SVD of orthogonal projection
    X_ortho = np.dot(joint_projection_ortho, X)
    U_xo, D_xo, V_xo = get_svd(X_ortho)

    # threshold singular values of orthogonal matrix
    individual_rank = sum(D_xo > sv_threshold)
    # TODO: what if individual_rank == 0?
    # TODO: give user option to manually select individual rank

    # estimate individual space
    I = svd_approx(U_xo, D_xo, V_xo, individual_rank)

    # estimate noise
    E = X - (J + I)

    # return {'J': J, 'I': I, 'E': E, 'individual_rank': individual_rank}
    return J, I, E, individual_rank
