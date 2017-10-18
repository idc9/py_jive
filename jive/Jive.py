from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from .lin_alg_fun import get_svd
from .JiveBlock import JiveBlock


class Jive(object):

    def __init__(self, blocks, init_svd_ranks=None, wedin_estimate=True,
                 save_full_final_decomp=True, show_scree_plot=True):
        """
        Paramters
        ---------
        Blocks: a list of data matrices

        init_svd_ranks: (optional) list of ranks of the first SVD for each data
        block -- should be larger than the signal rank.
        A value of None will compute the full SVD. Sparse data matrices require
        a value for init_svd_ranks. 

        wedin_estimate: use the wedin bound to estimate the joint space
        (True/False) if False then the user has to manually set the joint space
        rank

        save_full_final_decomp: whether or not to save the I, J, E final decomposition matrices

        show_scree_plot: show the scree plot of the initial SVD
        """
        # TODO: rename wedin_estimate to use_wedin_estimate
        self.K = len(blocks)  # number of blocks

        self.n = blocks[0].shape[0]  # number of observation
        for k in range(self.K):  # chack observation consistency
            if self.n != blocks[k].shape[0]:
                raise ValueError("Each block must have same number of observations (rows)")

        self.dimensions = [blocks[k].shape[1] for k in range(self.K)]

        if init_svd_ranks is None:  # Set every block to None by default
            init_svd_ranks = [None] * self.K
        elif len(init_svd_ranks) != self.K:
            raise ValueError("Must provide each block a value for init_svd_ranks (or set it to None)")

        # initialize blocks
        self.blocks = []
        for k in range(self.K):
            self.blocks.append(JiveBlock(blocks[k],
                                         init_svd_ranks[k],
                                         save_full_final_decomp,
                                         'block ' + str(k + 1)))

        # scree plot to decide on signal ranks
        if show_scree_plot:
            self.scree_plot()

        self.wedin_estimate = wedin_estimate

    def get_block_initial_singular_values(self):
        """
        Returns the singluar values for the initial SVD for each block.
        """
        # TODO: rename
        return [self.blocks[k].D for k in range(self.K)]

    def scree_plot(self, log=False, diff=False):
        """
        Draw scree plot for each data block
        """
        # scree plots for inital SVD
        plt.figure(figsize=[5 * self.K, 5])
        for k in range(self.K):
            plt.subplot(1, self.K, k + 1)
            self.blocks[k].scree_plot(log, diff)

    def set_signal_ranks(self, signal_ranks):
        """
        signal ranks is a list of length K
        """
        for k in range(self.K):
            self.blocks[k].set_signal_rank(signal_ranks[k])

        # possibly estimate joint space
        if self.wedin_estimate:
            self.estimate_joint_space_wedin_bound()

    def estimate_joint_space_wedin_bound(self):
        """
        Estimate joint score space and compute final decomposition
        - SVD on joint scores matrix
        - find joint rank using wedin bound threshold
        """

        # compute wedin bounds
        for k in range(self.K):
            self.blocks[k].compute_wedin_bound()

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
        self.joint_scores, self.joint_sv, self.joint_loadings =  get_svd(joint_scores_matrix)

        # estimate joint rank with wedin bound
        if self.K == 2:
            principal_angles = np.array([np.arccos(d ** 2 - 1) for d in self.joint_sv]) * (180.0/np.pi)
            self.joint_rank = sum(principal_angles < phi_est)
        else:
            self.joint_rank = sum(self.joint_sv ** 2 > joint_sv_bound)

        # select basis for joint space
        self.joint_scores = self.joint_scores[:, 0:self.joint_rank]
        self.joint_loadings = self.joint_loadings[:, 0:self.joint_rank]
        self.joint_sv = self.joint_sv[0:self.joint_rank]

        # possibly remove columns
        self.reconsider_joint_components()

        # can now compute final decomposotions
        self.compute_final_decomposition()

    def set_joint_rank(self, joint_rank):
        """
        Manualy set the joint space rank

        Paramters
        ---------
        joint_rank: user selected rank of the estimated joint space
        """

        self.joint_rank = joint_rank

        # TODO: this could be a K-SVD not a full SVD
        joint_scores_matrix = np.bmat([self.blocks[k].signal_basis for k in range(self.K)])
        self.joint_scores, self.joint_sv, self.joint_loadings =  get_svd(joint_scores_matrix)

        # select basis for joint space
        self.joint_scores = self.joint_scores[:, 0:self.joint_rank]
        self.joint_loadings = self.joint_loadings[:, 0:self.joint_rank]
        self.joint_sv = self.joint_sv[0:self.joint_rank]

        # TODO: add this as an option
        # # possibly remove some columns
        # if reconsider_joint_components:
        #     self.reconsider_joint_components()

        # can now compute final decomposotions
        self.compute_final_decomposition()

    def reconsider_joint_components(self):
        """
        Checks the identifability
        """
        # TODO: possibly make this a function outside the class

        # check identifiability constraint
        to_keep = set(range(self.joint_rank))
        for k in range(self.K):
            for j in range(self.joint_rank):
                # This might be joint_sv
                score = np.dot(self.blocks[k].X.T, self.joint_scores[:, j])
                sv = np.linalg.norm(score)

                # if sv is below the thrshold for any data block remove j
                if sv < self.blocks[k].sv_threshold:
                    # TODO: should probably keep track of this
                    print('removing column ' + str(j))
                    to_keep.remove(j)
                    break

        # remove columns of joint_scores that don't satisfy the constraint
        self.joint_rank = len(to_keep)
        self.joint_scores = self.joint_scores[:, list(to_keep)]
        self.joint_loadings = self.joint_loadings[:, list(to_keep)]
        self.joint_sv = self.joint_sv[list(to_keep)]

        if self.joint_rank == 0:
            # TODO: how to handle this situation?
            print('warning all joint signals removed')

    def compute_final_decomposition(self):
        # final decomposotion
        for k in range(self.K):
            self.blocks[k].final_decomposition(self.joint_scores)

    def get_jive_estimates(self):
        """
        Returns the jive decomposition for each data block. We can decomose
        the full data matix as

        X = J + I + E

        then decompose both J and I with an SVD

        J = U D V.T
        I = U D V.T

        Output
        ------
        a list of block JIVE estimates which have the following structure

        estimates[k]['individual']['full'] returns the full individual estimate
        for the kth data block (this is the I matrix). You can replace
        'individual' with 'joint'. Similarly you can replace 'full' with
        'scores', 'sing_vals', 'loadings', 'rank'
        """
        return [self.blocks[k].get_jive_estimates() for k in range(self.K)]

    def get_joint_space_estimate(self):
        """"
        Returns the SVD of the concatonated scores matrix.
        """
        return {'scores': self.joint_scores,
                'sing_vals': self.joint_sv,
                'loadings': self.joint_loadings,
                'rank': self.joint_rank}

    def get_block_estimates(self):
        """
        Returns the jive decomposition for each data block.

        Output
        ------
        a list of block JIVE estimates which have the following structure

        estimates[k]['individual']['full'] returns the full individual estimate
        for the kth data block (this is the I matrix). You can replace
        'individual' with 'joint'. Similarly you can replace 'full' with
        'scores', 'sing_vals', 'loadings', 'ranks'
        """
        return [self.blocks[k].get_jive_estimates() for k in range(self.K)]

    def get_block_estimates_full(self):
        """
        Returns the jive decomposition for each data block. Note of full=False
        you can only run this method onces because it will kill each X matrix.

        Output
        ------
        a list of the full block estimates (I, J, E) i.e. estimates[k]['J']
        """
        # TODO: give the option to return only some of I, J and E
        return [self.blocks[k].get_full_jive_estimates()
                for k in range(self.K)]
