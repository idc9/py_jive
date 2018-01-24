from __future__ import print_function

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import warnings

from jive.lin_alg_fun import svd_wrapper
from jive.JiveBlock import JiveBlock

from jive.diagnostic_plot import plot_joint_diagnostic

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

        # whether or not JIVE has computed the final decomposition
        self.has_finished = False


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

    def sample_wedin_bounds(self, num_samples=1000):
        """
        For each block, generate samples to estimate the wedin bound (described in section 2 of the AJIVE paper)
        """

        if not hasattr(self, 'initial_svs'):
            raise ValueError('Please run compute_initial_svd() before sampling for the wedin bound')

        for k in range(self.K):
            self.blocks[k].sample_wedin_bound(num_samples)

        self.wedin_sv_samples = [self.K - sum([min(self.blocks[k].wedin_samples[i], 1) ** 2
                                 for k in range(self.K)]) for i in range(num_samples)]

    def sample_random_direction_bounds(self, num_samples=1000):
        if not hasattr(self, 'initial_svs'):
            raise ValueError('Please run compute_initial_svd() before sampling the random direction bounds')

        self.random_sv_samples = [0.0]*num_samples
        signal_ranks = [self.blocks[k].signal_rank for k in range(self.K)]
        for i in range(num_samples):

            M = [0]*self.K
            for k in range(self.K):

                # sample random orthonormal basis
                Z = np.random.normal(size=[self.n, signal_ranks[k]])
                M[k] = np.linalg.qr(Z)[0]

            # compute largest sing val of random joint matrix
            M = np.bmat(M)
            _, svs, __ = svd_wrapper(M)
            self.random_sv_samples[i] = max(svs) ** 2


    # def compute_wedin_bound(self,
    #                         sampling_procedures=None,
    #                         num_samples=1000,
    #                         quantile='median',
    #                         qr=True):
    #     # TODO: remove this
    #     """
    #     Estimate joint score space and compute final decomposition
    #     - SVD on joint scores matrix
    #     - find joint rank using wedin bound threshold


    #     Parameters
    #     ----------
    #     sampling_procedures: which sampling procedure each block should use. Can
    #     be either None or list with entries either 'svec_resampling' or 'sample_project'.
    #     If None the will defer use svec_resampling for dense matrices and sample_project for sparse matrices.

    #     num_samples: number of columns to resample for wedin bound

    #     quantile: for wedin bound TODO better description
    #     """

    #     if not hasattr(self, 'joint_scores'):
    #         raise ValueError('Please run compute_joint_svd() before computing the wedin bound')

    #     # TODO: think more about behaviour of sampling_procedures
    #     if sampling_procedures is None:
    #         sampling_procedures = [None] * self.K

    #     # compute wedin bound for each block
    #     for k in range(self.K):
    #         self.blocks[k].compute_wedin_bound(sampling_procedures[k], num_samples, quantile, qr)

    #     sin_bound_ests = [self.blocks[k].sin_bound_est for k in range(self.K)]

    #     # compute theshold and count how many singular values are above the threshold
    #     # TODO: double check we want min(b, 1)
    #     wedin_threshold = self.K - sum([min(b, 1) ** 2 for b in sin_bound_ests])
    #     joint_rank_wedin_estimate = sum(self.joint_sv ** 2 > wedin_threshold)

    #     # if JIVE thinks everything is in the joint space i.e.
    #     # the joint rank is equal to the sum of the signal ranks
    #     if joint_rank_wedin_estimate == self.total_signal_dim:
    #         warnings.warn('The wedin bound estimate thinks the entire signal space is joint. This could mean the wedin bound is too weak.')

    #     self.sin_bound_ests = sin_bound_ests
    #     self.joint_rank_wedin_estimate = joint_rank_wedin_estimate

    def compute_joint_svd(self):

        # SVD on joint scores matrx
        joint_scores_matrix = np.bmat([self.blocks[k].signal_basis for k in range(self.K)])
        self.total_signal_dim = joint_scores_matrix.shape[1] # TODO: maybe rename this

        self.joint_scores, self.joint_sv, self.joint_loadings =  svd_wrapper(joint_scores_matrix)

    def estimate_joint_rank(self, num_samples=1000,
                            wedin_percentile=95, random_percentile=5):
        """
        Estimates the joint rank using the wedin and random sampling bounds

        Parameters
        ----------
        num_samples:

        wedin_percentile:

        random_percentile:
        """

        if not hasattr(self, 'joint_sv'):
            raise ValueError('Please run compute_joint_svd() before making diagnostic plot')

        self.sample_wedin_bounds(num_samples)
        self.sample_random_direction_bounds(num_samples)

        wedin_cutoff = np.percentile(self.wedin_sv_samples, wedin_percentile)
        random_cutoff = np.percentile(self.random_sv_samples, random_percentile)
        svsq_cutoff = max(wedin_cutoff, wedin_cutoff)

        self.joint_rank_estimate = sum(self.joint_sv ** 2 > svsq_cutoff)

    def plot_joint_diagnostic(self, wedin_percentile=95, random_percentile=5):
        """
        Plots joint rank threshold diagnostic plot
        """

        if not hasattr(self, 'wedin_sv_samples'):
            raise ValueError('Please run sample_wedin_bounds() before making diagnostic plot')

        if not hasattr(self, 'random_sv_samples'):
            raise ValueError('Please run sample_random_direction_bounds() before making diagnostic plot')

        if not hasattr(self, 'joint_sv'):
            raise ValueError('Please run compute_joint_svd() before making diagnostic plot')

        plot_joint_diagnostic(joint_svsq=self.joint_sv ** 2,
                              wedin_sv_samples=self.wedin_sv_samples,
                              random_sv_samples=self.random_sv_samples,
                              wedin_percentile=wedin_percentile,
                              random_percentile=random_percentile)

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
            # maybe do nothing
            print('warning all joint signals removed!')

    def compute_block_specific_spaces(self,
                                      save_full_estimate=True,
                                      individual_ranks=None):
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

        self.has_finished = True

    def estimate_jive_spaces(self,
                             reconsider_joint_components=True,
                             save_full_estimate=True,
                             num_samples=1000,
                             wedin_percentile=95,
                             random_percentile=5):
        """
        Computes wedin bound, set's joint rank from wedin bound estimate, then
        computes final decomposition

        Parameters
        ----------
        reconsider_joint_components: whether or not to remove columns not satisfying
        identifiability constraint`

        save_full_estimate: whether or not to save the full I, J, E matrices

        num_samples: number of columns to resample for wedin bound

        """
        self.compute_joint_svd()

        self.estimate_joint_rank(num_samples=num_samples,
                                 wedin_percentile=wedin_percentile,
                                 random_percentile=random_percentile)


        # if JIVE thinks everything is in the joint space i.e.
        # the joint rank is equal to the sum of the signal ranks
        if self.joint_rank_estimate == self.total_signal_dim:
            warnings.warn('The wedin + random bound estimate thinks the entire signal space is joint. This could mean the wedin bound is too weak.')

        self.set_joint_rank(self.joint_rank_estimate,
                            reconsider_joint_components)

        self.compute_block_specific_spaces(save_full_estimate)

        self.has_finished = True


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
        'scores', 'sing_vals', 'loadings', or 'rank'
        """

        if not self.has_finished:
            raise ValueError('JIVE has not yet computed block decomposition e.g. run estimate_jive_spaces()')

        return [self.blocks[k].get_block_estimates() for k in range(self.K)]

    def get_common_joint_space_estimate(self):
        """"
        Returns the SVD of the concatonated scores matrix.
        """
        if not hasattr(self, 'joint_scores') :
            raise ValueError('joints space estimation has not yet been computed')

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
        if not self.has_finished:
            raise ValueError('JIVE has not yet computed block decomposition e.g. run estimate_jive_spaces()')

        # TODO: give the option to return only some of I, J and E
        return [self.blocks[k].get_full_estimates()
                for k in range(self.K)]

    def save_estimates(self, fname='', notes='', force=False):
        """
        Saves the JIVE estimates

        U, D, V, full, rank for block secific joint/individual spaces
        U, D, V, rank for common joint space
        some metadata (when saved, some nots)

        Parameters
        ----------
        fname: name of the file
        notes: any notes you want to include
        force: whether or note to overwrite a file with the same name
        """
        if not self.has_finished:
            raise ValueError('JIVE has not yet computed block decomposition e.g. run estimate_jive_spaces()')

        if os.path.exists(fname) and (not force):
            raise ValueError('%s already exists' % fname)

        kwargs = {}
        svd_dat = ['scores', 'sing_vals', 'loadings', 'rank']
        kwargs['K'] = self.K

        block_estimates = self.get_block_specific_estimates()
        for k in range(self.K):
            for mode in ['joint', 'individual']:
                for dat in svd_dat + ['full']:
                    label = '%d_%s_%s' % (k, mode, dat)
                    kwargs[label] = block_estimates[k][mode][dat]

        common_joint = self.get_common_joint_space_estimate()
        for dat in svd_dat:
            kwargs['common_%s' % dat] = common_joint[dat]

        current_time = time.strftime("%m/%d/%Y %H:%M:%S")
        kwargs['metadata'] = [current_time, notes]

        np.savez_compressed(fname, **kwargs)


    def save_init_svd(self, fname='', notes='', force=False):
        """
        Saves the initial SVD so it can be loaded later without recomputing

        Parameters
        ----------
        fname: name of the file
        notes: any notes you want to include
        force: whether or note to overwrite a file with the same name
        """

        if not hasattr(self.blocks[0], 'scores'):
            raise ValueError('initial svd has not yet been computed')

        if os.path.exists(fname) and (not force):
            raise ValueError('%s already exists' % fname)

        kwargs = {}
        svd_dat = ['scores', 'sing_vals', 'loadings', 'rank']
        kwargs['K'] = self.K

        for k in range(self.K):
            kwargs['%d_scores' % k] = self.blocks[k].scores
            kwargs['%d_sv' % k] = self.blocks[k].sv
            kwargs['%d_loadings' % k ] = self.blocks[k].loadings
            kwargs['%d_init_svd_rank' % k] = self.blocks[k].init_svd_rank

        np.savez_compressed(fname, **kwargs)

    def init_svd_from_saved(self, fname):
        """
        Loads the initial SVD from a saved file

        Parameters
        ----------
        fname: path to file saved with save_init_svd
        """
        saved_data = np.load(fname)
        K = saved_data['K']

        for k in range(K):
            self.blocks[k].scores = saved_data['%d_scores' % k]
            self.blocks[k].sv = saved_data['%d_sv' % k]
            self.blocks[k].loadings = saved_data['%d_loadings' % k]
            self.blocks[k].init_svd_rank = saved_data['%d_init_svd_rank' % k]


def get_saved_jive_estimates(fname=''):
    """
    Returns JIVE estimates that have been saved to disk.

    Parameters
    ----------
    fname: name of file saved with jive.save_estimates()

    Output
    ------
    block_estimates, common_joint_estimates, metadata

    block_estimates: block specific estimates
    common_joint_estimates: common joint space
    metadata: time of save and notes
    """
    saved_data = np.load(fname)

    K = saved_data['K']
    svd_dat = ['scores', 'sing_vals', 'loadings', 'rank']
    modes = ['joint', 'individual']

    block_estimates = [{mode: {dat: [] for dat in svd_dat + ['full']}
                        for mode in modes} for _ in range(K)]

    for k in range(K):
        for mode in modes:
            for dat in svd_dat + ['full']:
                label = '%d_%s_%s' % (k, mode, dat)
                block_estimates[k][mode][dat] = saved_data[label]

    common_joint_estimates = {dat : [] for dat in svd_dat}
    for dat in svd_dat:
        common_joint_estimates[dat] = saved_data['common_%s' % dat]

    metadata = saved_data['metadata']

    return block_estimates, common_joint_estimates, metadata

