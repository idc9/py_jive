from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt

from jive.lin_alg_fun import svd_wrapper
from jive.JiveBlock import JiveBlock


class Jive(object):

    def __init__(self, blocks):
        """
        Paramters
        ---------
        Blocks: a list of data matrices

        """
        self.K = len(blocks)  # number of blocks

        # TODO: maybe convert blocks to ndarray

        self.n = blocks[0].shape[0]  # number of observation
        for k in range(self.K):  # chack observation consistency
            if self.n != blocks[k].shape[0]:
                raise ValueError("Each block must have same number of observations (rows)")

        self.dimensions = [blocks[k].shape[1] for k in range(self.K)]

        # initialize blocks
        self.blocks = []
        for k in range(self.K):
            self.blocks.append(JiveBlock(blocks[k], 'block ' + str(k + 1)))


    def compute_initial_svd(self, init_svd_ranks=None):
        """
        Compute initial SVD for each block

        init_svd_ranks: list of ranks of the first SVD for each data
        block -- should be larger than the signal rank.
        A value of None will compute the full SVD. Sparse data matrices require
        a value for init_svd_ranks, otherwise this is optional.

        """

        if init_svd_ranks is None:
            init_svd_ranks = [None] * self.K

        for k in range(self.K):
            self.blocks[k].initial_svd(init_svd_ranks[k])

        # stores initial svs for each block
        self.initial_svs = [self.blocks[k].sv for k in range(self.K)]

    def scree_plots(self, log=False, diff=False):
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


    def compute_joint_svd(self):
        
        # SVD on joint scores matrx
        joint_scores_matrix = np.bmat([self.blocks[k].signal_basis for k in range(self.K)])
        self.joint_scores, self.joint_sv, self.joint_loadings =  svd_wrapper(joint_scores_matrix)


    def compute_wedin_bound(self, sampling_procedures=None, num_samples=1000, quantile='median'):
        """
        Estimate joint score space and compute final decomposition
        - SVD on joint scores matrix
        - find joint rank using wedin bound threshold


        Parameters
        ----------
        sampling_procedures: which sampling procedure each block should use. Can
        be either None or list with entries either 'svec_resampling' or 'sample_project'. 
        If None the will defer use svec_resampling for dense matrices and sample_project for sparse matrices.

        num_samples: number of columns to resample for wedin bound

        quantile: for wedin bound TODO better description
        """

        # TODO: think more about behaviour of sampling_procedures

        if sampling_procedures is None:
            sampling_procedures = [None] * self.K

        # compute wedin bound for each block
        for k in range(self.K):
            self.blocks[k].compute_wedin_bound(sampling_procedures[k], num_samples, quantile)

        wedin_bounds = [self.blocks[k].wedin_bound for k in range(self.K)]


        # TODO: can probbaly kill K=2 case

        # threshold for joint space segmentaion
        if self.K == 2:  # if two blocks use angles
            theta_est_1 = np.arcsin(min(wedin_bounds[0], 1))
            theta_est_2 = np.arcsin(min(wedin_bounds[1], 1))
            phi_est = np.sin(theta_est_1 + theta_est_2) * (180.0/np.pi)
        else:
            joint_sv_bound = self.K - sum([b ** 2 for b in wedin_bounds])


        # estimate joint rank with wedin bound
        if self.K == 2:
            principal_angles = np.array([np.arccos(d ** 2 - 1) for d in self.joint_sv]) * (180.0/np.pi)
            joint_rank_wedin_estimate = sum(principal_angles < phi_est)
        else:
            joint_rank_wedin_estimate = sum(self.joint_sv ** 2 > joint_sv_bound)


        self.wedin_bounds = wedin_bounds
        self.joint_rank_wedin_estimate = joint_rank_wedin_estimate


    def set_joint_rank(self, joint_rank, reconsider_joint_components=False):
        """
        Sets the joint rank

        Paramters
        ---------
        joint_rank: user selected rank of the estimated joint space

        reconsider_joint_components: whether or not to remove columns not satisfying
        identifiability constraint
        """

        if not hasattr(self, 'joint_scores'):
            raise ValueError('please run compute_joint_svd before setting joint rank')

        self.joint_rank = joint_rank

        # select basis for joint space
        self.joint_scores = self.joint_scores[:, 0:self.joint_rank]
        self.joint_loadings = self.joint_loadings[:, 0:self.joint_rank]
        self.joint_sv = self.joint_sv[0:self.joint_rank]

        # possibly remove some columns
        if reconsider_joint_components:
            self.reconsider_joint_components()

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
            print('warning all joint signals removed!')

    def compute_block_specific_spaces(self, save_full_estimate=False, individual_ranks=None):
        """
        Computes final decomposition and estimates for block specific
        joint and individual space

        Parameters
        ----------
        save_full_estimate: whether or not to save the full I, J, E matrices

        individual_ranks: gives user option to specify individual ranks. 
        Either a list of length K or None. Setting an entry equal to None
        will cause that block to estimate its individual rank.

        """


        if individual_ranks is None:
            individual_ranks = [None] * self.K
        elif len(individual_ranks) != self.K:
            raise ValueError("Must provide each block a value for individual_ranks (or set it to None)")

        if not hasattr(self, 'joint_scores'):
            raise ValueError('please run compute_joint_svd')

        if not hasattr(self, 'joint_rank'):
            raise ValueError('please set the joint rank.')

        # final decomposotion
        for k in range(self.K):
            self.blocks[k].compute_final_decomposition(self.joint_scores, individual_ranks[k], save_full_estimate)


    def estimate_jive_spaces_wedin_bound(self,
                                         reconsider_joint_components=True,
                                         save_full_estimate=False,
                                         sampling_procedures=None,
                                         num_samples=1000,
                                         quantile='median'):
        """ 
        Computes wedin bound, set's joint rank from wedin bound estimate, then
        computes final decomposition

        Parameters
        ----------
        reconsider_joint_components: whether or not to remove columns not satisfying
        identifiability constraint`

        save_full_estimate: whether or not to save the full I, J, E matrices

        sampling_procedures: which sampling procedure each block should use. Can
        be either None or list with entries either 'svec_resampling' or 'sample_project'
        If None the will defer use svec_resampling for dense matrices and sample_project for sparse matrices.

        num_samples: number of columns to resample for wedin bound

        quantile: for wedin bound TODO better description
        """
        self.compute_joint_svd()

        self.compute_wedin_bound(sampling_procedures, num_samples, quantile)

        self.set_joint_rank(self.joint_rank_wedin_estimate,
                            reconsider_joint_components)

        self.compute_block_specific_spaces(save_full_estimate)


    def get_block_specific_estimates(self):
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

        return [self.blocks[k].get_block_estimates() for k in range(self.K)]

    def get_common_joint_space_estimate(self):
        """"
        Returns the SVD of the concatonated scores matrix.
        """
        return {'scores': self.joint_scores,
                'sing_vals': self.joint_sv,
                'loadings': self.joint_loadings,
                'rank': self.joint_rank}


    def get_block_full_estimates(self):
        """
        Returns the jive decomposition for each data block. Note of full=False
        you can only run this method onces because it will kill each X matrix.

        Output
        ------
        a list of the full block estimates (I, J, E) i.e. estimates[k]['J']
        """
        # TODO: give the option to return only some of I, J and E
        return [self.blocks[k].get_full_estimates()
                for k in range(self.K)]
