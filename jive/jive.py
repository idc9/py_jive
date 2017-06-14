import numpy as np
from lin_alg_fun import *


class Jive(object):

    def __init__(self, blocks):
        """
        Paramters
        ---------
        Blocks is a list of data matrices
        """
        self.K = len(blocks)  # number of blocks

        self.n = blocks[0].shape[0]  # number of data points

        # initialize blocks
        self.blocks = []
        for k in range(self.K):
            self.blocks.append(Block(blocks[k], 'block ' + str(k)))

        # scree plot to decide on signal ranks
        self.scree_plot()

    def scree_plot(self):
        """
        Draw scree plot for each data block
        """
        # scree plots for inital SVD
        plt.figure(figsize=[5 * self.K, 5])
        for k in range(self.K):
            plt.subplot(1, self.K, k + 1)
            self.blocks[k].scree_plot()

    def set_signal_ranks(self, signal_ranks):
        """
        signal ranks is a list of length K
        """
        for k in range(self.K):
            self.blocks[k].set_signal_rank(signal_ranks[k])

        # can now compute joint space then final decomposition
        self.score_space_segmentation_and_final_decomposition()

    def score_space_segmentation_and_final_decomposition(self):
        """
        Estimate joint score space and compute final decomposition
        - SVD on joint scores matrix
        - find joint rank using wedin bound threshold
        - estimate J, I, E for each block
        """

        # wedin bound estimates
        wedin_bounds = [self.blocks[k].wedin_bound for k in range(self.K)]

        # threshold for joint space segmentaion
        if self.K == 2:  # if two blocks use angles
            theta_est_1 = np.arcsin(min(wedin_bounds[0], 1))
            theta_est_2 = np.arcsin(min(wedin_bounds[1], 1))
            phi_est = np.sin(theta_est_1 + theta_est_2) * (180.0/np.pi)
        else:
            joint_sv_bound = self.K - sum([b ** 2 for b in wedin_bounds])

        # SVD on joint scores matrx
        joint_scores_matrix = np.bmat([self.blocks[k].signal_basis for k in range(self.K)])
        U_joint, D_joint, V_joint = get_svd(joint_scores_matrix)

        # estimate joint rank with wedin bound
        if self.K == 2:
            principal_angles = np.array([np.arccos(d ** 2 - 1) for d in D_joint]) * (180.0/np.pi)
            joint_rank = sum(principal_angles < phi_est)
        else:
            joint_rank = sum(D_joint ** 2 > joint_sv_bound)

        # select basis for joint space
        joint_scores = U_joint[:, 0:joint_rank]

        # check identifiability constraint
        to_keep = set(range(joint_rank))
        for k in range(self.K):
            for j in range(joint_rank):
                score = np.dot(self.blocks[k].X.T, joint_scores[:, j])
                sv = np.linalg.norm(score)

                # if sv is below the thrshold for any data block remove j
                if sv < self.blocks[k].sv_threshold:
                    # TODO: should probably keep track of this
                    print 'removing column ' + str(j)
                    to_keep.remove(j)
                    break

        # remove columns of joint_scores that don't satisfy the constraint
        joint_scores = joint_scores[:, list(to_keep)]
        joint_rank = len(to_keep)
        self.joint_rank = joint_rank

        if joint_rank == 0:
            # TODO: how to handle this situation?
            print 'warning all joint signals removed'

        # final decomposotion
        joint_projection = projection_matrix(joint_scores)
        for k in range(self.K):
            self.blocks[k].final_decomposition(joint_projection)

    def get_jive_estimates(self):
        """
        Returns the jive decomposition for each data block.

        Output
        ------
        a list of block JIVE estimates i.e. estimates[k]['J'] gives the
        estimated J matrix for the kth block
        """
        estimates = {}
        for k in range(self.K):
            estimates[k] = {'J': self.blocks[k].J,
                            'I': self.blocks[k].I,
                            'E': self.blocks[k].E,
                            'individual_rank': self.blocks[k].individual_rank}

        estimates['joint_rank'] = self.joint_rank

        return estimates


class Block(object):

    def __init__(self, X, name=None):
        """
        Stores a single data block (rows as observations).

        Paramters
        ---------
        X: a data block

        name: the (optional) name of the block
        """

        self.name = name

        self.X = X
        self.n = X.shape[0]
        self.d = X.shape[1]

        # SVD for initial signal space extraction
        self.U, self.D, self.V = get_svd(X)

    def scree_plot(self):
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
        # self.U = None
        # self.V = None
        # self.D = None

    def final_decomposition(self, joint_projection):

        self.J, self.I, self.E, self.individual_rank = block_JIVE_decomposition(X=self.X,
                                                                                joint_projection=joint_projection,
                                                                                sv_threshold=self.sv_threshold)


def get_sv_threshold(singular_values, rank):
    """
    Returns the singular value threshold value for rank R; half way between
    the thresholding value and the next smallest.

    Paramters
    ---------
    singular_values: list of singular values

    rank: rank of the threshold
    """
    return .5 * (singular_values[rank - 1] + singular_values[rank])


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

    rs_norms = [0]*num_samples

    for s in range(num_samples):

        # sample columns from orthogonal basis
        sampled_col_index = np.random.choice(orthogonal_basis.shape[1], size=rank, replace=True)
        resampled_basis = orthogonal_basis[:, sampled_col_index]  # this is V* from AJIVE p12

        # project observed data
        if right_vectors:
            resampled_projection = np.dot(X, resampled_basis)
        else:
            resampled_projection = np.dot(X.T, resampled_basis)

        # compute resampled operator L2 nrm
        rs_norms[s] = np.linalg.norm(resampled_projection, ord=2)  # operator L2 norm

    return rs_norms


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
    # maybe this way??
    # EV_estimate = np.median(V_sampled_norms)
    # UE_estimate = np.median(U_sampled_norms)
    # wedin_bound_est = max(EV_estimate, UE_estimate)/sigma_min
    sigma_min = D[:rank]
    wedin_bound_samples = [max(U_sampled_norms[s], V_sampled_norms[s])/sigma_min for s in range(num_samples)]
    wedin_bound_est = np.median(wedin_bound_samples)

    return wedin_bound_est


def block_JIVE_decomposition(X, joint_projection, sv_threshold):
    """
    Computes the JIVE decomposition for an individual data block

    Parameters
    ----------
    X: observed data matrix

    joint_projection: projection matrix onto joint score space

    sv_threshold: singular value threshold

    Output
    ------
    J, I, E, individual_rank
    J: estimated joint signal matrix
    I: estimated indiviual signal
    E: esimated error matrix
    individual_rank: rank of the esimated individual space
    """

    # compute orthogonal projection matrix
    joint_projection_ortho = np.eye(X.shape[0]) - joint_projection

    # estimate joint spaces
    J = np.dot(joint_projection, X)

    # SVD of orthogonal projection
    X_ortho = np.dot(joint_projection_ortho, X)
    U_xo, D_xo, V_xo = get_svd(X_ortho)

    # threshold singular values of orthogonal matrix
    individual_rank = sum(D_xo > sv_threshold)

    # estimate individual space
    I = svd_approx(U_xo, D_xo, V_xo, individual_rank)

    # estimate noise
    E = X - (J + I)

    # return {'J': J, 'I': I, 'E': E, 'individual_rank': individual_rank}
    return J, I, E, individual_rank
