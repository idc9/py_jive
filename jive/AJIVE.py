import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy.sparse import issparse
from copy import deepcopy
from joblib import load, dump
import pandas as pd
from sklearn.linear_model import LinearRegression

from jive.utils import svd_wrapper, centering
from jive.lazymatpy.templates.matrix_transformations import col_proj, col_proj_orthog
from jive.wedin_bound import get_wedin_samples
from jive.random_direction import sample_randdir
from jive.viz.diagnostic_plot import plot_joint_diagnostic
from jive.viz.viz import plot_loading
from jive.PCA import PCA, get_comp_names


# TODO: give user option to store initial SVD
# TODO: common loadings direct regression
class AJIVE(object):
    """
    Angle-based Joint and Individual Variation Explained

    Parameters
    ----------
    init_signal_ranks: {list, dict}
        The initial signal ranks.

    joint_rank: {None, int}
        If None, will estimate the joint rank, otherwise will use
        provided joint rank.

    indiv_ranks: {list, dict, None}
        If None, will estimate the individual ranks. Otherwise will
        use provided individual ranks.

    center: {str, None}
        How to center the data matrices. If None, will not center.

    reconsider_joint_components: bool
        TODO: explain

    wedin_percentile: int (default=5)
        Percentile for wedin (lower) bound cutoff for squared singular values
        used to estimate joint rank.

    n_wedin_samples: int (default=1000)
        Number of wedin bound samples to draw.

    precomp_wedin_samples {None, dict of array-like, list of array-like}
        Precomputed Wedin samples for each data block.

    randdir_percentile: int (default=95)
        Percentile for random direction (lower) bound cutoff for squared
        singular values used to estimate joint rank..

    n_randdir_samples: int (default=1000)
        Number of random directions samples to draw.

    precomp_randdir_samples {None,  array-like}
        Precomputed random direction samples.

    n_jobs: int, None
        Number of jobs for parallel processing wedin samples and random
        direction samples using sklearn.externals.joblib.Parallel.
        If None, will not use parallel processing.

    Attributes
    ----------

    common: jive.PCA.PCA
        The common joint space.

    blocks: dict of BlockSpecificResults
        The block specific results.

    centers_: dict
        The the centering vectors computed for each matrix.


    sv_threshold_: dict
        The singular value thresholds computed for each block based on the
        initial SVD. Used to estimate the individual ranks.

    all_joint_svals_: list
        All singular values from the concatenated joint matrix.

    random_sv_samples_: list of floats
        Random singular value samples from for random direction bound.

    rand_cutoff_: float
        Singular value squared cutoff for the random direction bound.

    wedin_samples_: dict of lists of floats
        The wedin samples for each block.

    wedin_cutoff_: float
        Singular value squared cutoff for the wedin bound.

    svalsq_cutoff_: float
        max(rand_cutoff_, wedin_cutoff_)

    joint_rank_wedin_est_: int
        The joint rank estimated using the wedin/random direction bound.

    joint_rank: int
        The estimated joint rank.

    indiv_ranks: dict of ints
        The individual ranks for each block.

    zero_index_names: bool
        Whether or not to zero index component names.
    """

    def __init__(self,
                 init_signal_ranks,
                 joint_rank=None, indiv_ranks=None,
                 center=True,
                 reconsider_joint_components=True,
                 wedin_percentile=5, n_wedin_samples=1000,
                 precomp_wedin_samples=None,
                 randdir_percentile=95, n_randdir_samples=1000,
                 precomp_randdir_samples=None,
                 store_full=True, n_jobs=None, zero_index_names=True):

        self.init_signal_ranks = init_signal_ranks
        self.joint_rank = joint_rank
        self.indiv_ranks = indiv_ranks

        self.center = center

        self.wedin_percentile = wedin_percentile
        self.n_wedin_samples = n_wedin_samples
        self.wedin_samples_ = precomp_wedin_samples
        if precomp_wedin_samples is not None:
            self.n_wedin_samples = len(list(precomp_wedin_samples.values())[0])

        self.randdir_percentile = randdir_percentile
        self.n_randdir_samples = n_randdir_samples
        self.random_sv_samples_ = precomp_randdir_samples
        if precomp_randdir_samples is not None:
            self.n_randdir_samples = len(precomp_randdir_samples)

        self.reconsider_joint_components = reconsider_joint_components

        self.store_full = store_full

        self.n_jobs = n_jobs
        self.zero_index_names = zero_index_names

    def __repr__(self):
        # return 'JIVE'
        if self.is_fit:
            r = 'AJIVE, joint rank: {}'.format(self.common.rank)
            for bn in self.block_names:
                indiv_rank = self.blocks[bn].individual.rank
                r += ', block {} indiv rank: {}'.format(bn, indiv_rank)
            return r

        else:
            return 'AJIVE object, nothing computed yet'

    def fit(self, blocks, precomp_init_svd=None):
        """
        Fits the AJIVE decomposition.

        Parameters
        ----------
        blocks: list, dict
            The data matrices. If dict, will name blocks by keys, otherwise
            blocks are named by 0, 1, ...K. Data matrices must have observations
            on the rows and have the same number of observations i.e. the
            kth data matrix is shape (n_samples, n_features[k]).

        precomp_init_svd: {list, dict, None}, optional
            Precomputed initial SVD. Must have one entry for each data block.
            The SVD should be a 3 tuple (scores, svals, loadings), see output
            of jive.utils.svd_wrapper for formatting details.

        """
        blocks, self.init_signal_ranks, self.indiv_ranks, precomp_init_svd,\
            self.center, obs_names, var_names, self.shapes_ = \
                arg_checker(blocks,
                            self.init_signal_ranks,
                            self.joint_rank,
                            self.indiv_ranks,
                            precomp_init_svd,
                            self.center)

        block_names = list(blocks.keys())
        num_obs = list(blocks.values())[0].shape[0]

        # center blocks
        self.centers_ = {}
        for bn in block_names:
            blocks[bn], self.centers_[bn] = centering(blocks[bn],
                                                      method=self.center[bn])

        ################################################################
        # step 1: initial signal space extraction by SVD on each block #
        ################################################################

        init_signal_svd = {}
        self.sv_threshold_ = {}
        for bn in block_names:

            # compute rank init_signal_ranks[bn] + 1 SVD of the data block
            if precomp_init_svd[bn] is None:
                # signal rank + 1 to get individual rank sv threshold
                U, D, V = svd_wrapper(blocks[bn], self.init_signal_ranks[bn] + 1)
            else:
                U = precomp_init_svd[bn]['scores']
                D = precomp_init_svd[bn]['svals']
                V = precomp_init_svd[bn]['loadings']

            # The SV threshold is halfway between the init_signal_ranks[bn]th
            # and init_signal_ranks[bn] + 1 st singular value. Recall that
            # python is zero indexed.
            self.sv_threshold_[bn] = (D[self.init_signal_ranks[bn] - 1] \
                                      + D[self.init_signal_ranks[bn]]) / 2

            init_signal_svd[bn] = {'scores': U[:, 0:self.init_signal_ranks[bn]],
                                   'svals': D[0:self.init_signal_ranks[bn]],
                                   'loadings': V[:, 0:self.init_signal_ranks[bn]]}

        ##################################
        # step 2: joint space estimation #
        ##################################
        # this step estimates the joint rank and computes the common
        # joint space basis

        # SVD of joint signal matrix
        joint_scores_matrix = np.bmat([init_signal_svd[bn]['scores'] for bn in block_names])
        joint_scores, joint_svals, joint_loadings = svd_wrapper(joint_scores_matrix)
        self.all_joint_svals_ = deepcopy(joint_svals)

        # estimate joint rank using wedin bound and random direction if a
        # joint rank estimate has not already been provided
        # TODO: maybe make this into an object or function
        if self.joint_rank is None:

            # if the random sv samples are not already provided compute them
            if self.random_sv_samples_ is None:
                self.random_sv_samples_ = \
                    sample_randdir(num_obs,
                                   signal_ranks=list(self.init_signal_ranks.values()),
                                   R=self.n_randdir_samples,
                                   n_jobs=self.n_jobs)

            # if the wedin samples are not already provided compute them
            if self.wedin_samples_ is None:
                self.wedin_samples_ = {}
                for bn in block_names:
                    self.wedin_samples_[bn] = \
                        get_wedin_samples(X=blocks[bn],
                                          U=init_signal_svd[bn]['scores'],
                                          D=init_signal_svd[bn]['svals'],
                                          V=init_signal_svd[bn]['loadings'],
                                          rank=self.init_signal_ranks[bn],
                                          R=self.n_wedin_samples,
                                          n_jobs=self.n_jobs)

            self.wedin_sv_samples_ = len(blocks) - \
                np.array([sum(self.wedin_samples_[bn][i] ** 2 for bn in block_names)
                          for i in range(self.n_wedin_samples)])

            # given the wedin and random bound samples, compute the joint rank
            # SV cutoff
            self.wedin_cutoff_ = np.percentile(self.wedin_sv_samples_,
                                               self.wedin_percentile)
            self.rand_cutoff_ = np.percentile(self.random_sv_samples_,
                                              self.randdir_percentile)
            self.svalsq_cutoff_ = max(self.wedin_cutoff_, self.rand_cutoff_)
            self.joint_rank_wedin_est_ = sum(joint_svals ** 2 > self.svalsq_cutoff_)
            self.joint_rank = deepcopy(self.joint_rank_wedin_est_)

        # check identifiability constraint and possibly remove some
        # joint components
        if self.reconsider_joint_components:
            joint_scores, joint_svals, joint_loadings, self.joint_rank = \
                reconsider_joint_components(blocks, self.sv_threshold_,
                                            joint_scores, joint_svals, joint_loadings,
                                            self.joint_rank)

        # TODO: include center?
        # TODO: comp_names, var_names
        # The common joint space has now been estimated
        self.common = PCA.from_precomputed(scores=joint_scores[:, 0:self.joint_rank],
                                           svals=joint_svals[0:self.joint_rank],
                                           loadings=joint_loadings[:, 0:self.joint_rank],
                                           obs_names=obs_names)

        self.common.set_comp_names(base='common',
                                   zero_index=self.zero_index_names)

        #######################################
        # step 3: compute final decomposition #
        #######################################
        # this step computes the block specific estimates

        block_specific = {bn: {} for bn in block_names}
        for bn in block_names:
            X = blocks[bn]

            ########################################
            # step 3.1: block specific joint space #
            ########################################
            # project X onto the joint space then compute SVD
            if self.joint_rank != 0:
                if issparse(X):  # lazy evaluation for sparse matrices
                    J = col_proj(X, joint_scores)
                    U, D, V = svd_wrapper(J, self.joint_rank)
                    J = None  # kill J matrix to save memory

                else:
                    J = np.array(np.dot(joint_scores, np.dot(joint_scores.T, X)))
                    U, D, V = svd_wrapper(J, self.joint_rank)
                    if not self.store_full:
                        J = None  # kill J matrix to save memory

            else:
                U, D, V = None, None, None
                if self.store_full:
                    J = np.zeros(shape=blocks[bn].shape)
                else:
                    J = None

            block_specific[bn]['joint'] = {'full': J,
                                           'scores': U,
                                           'svals': D,
                                           'loadings': V,
                                           'rank': self.joint_rank}

            #############################################
            # step 3.2: block specific individual space #
            #############################################
            # project X onto the orthogonal complement of the joint space,
            # estimate the individual rank, then compute SVD

            if issparse(X):  # lazy evaluation for sparse matrices
                U, D, V, indiv_rank = indiv_space_for_sparse(X,
                                                             joint_scores,
                                                             self.joint_rank,
                                                             self.init_signal_ranks[bn],
                                                             self.sv_threshold_[bn])
                I = None

            else:

                # project X columns onto orthogonal complement of joint_scores
                if self.joint_rank == 0:
                    X_orthog = X
                else:
                    X_orthog = X - np.dot(joint_scores,
                                          np.dot(joint_scores.T, X))

                # estimate individual rank using sv threshold, then compute SVD
                if self.indiv_ranks[bn] is None:
                    max_rank = min(X.shape) - self.joint_rank  # saves computation
                    U, D, V = svd_wrapper(X_orthog, max_rank)
                    rank = sum(D > self.sv_threshold_[bn])

                    if rank == 0:
                        U, D, V = None, None, None
                    else:
                        U = U[:, 0:rank]
                        D = D[0:rank]
                        V = V[:, 0:rank]

                    self.indiv_ranks[bn] = rank

                else:  # indiv_rank has been provided by the user
                    rank = self.indiv_ranks[bn]
                    if rank == 0:
                        U, D, V = None, None, None
                    else:
                        U, D, V = svd_wrapper(X_orthog, rank)

                if self.store_full:
                    if rank == 0:
                        I = np.zeros(shape=blocks[bn].shape)
                    else:
                        I = np.array(np.dot(U, np.dot(np.diag(D), V.T)))
                else:
                    I = None  # Kill I matrix to save memory

            block_specific[bn]['individual'] = {'full': I,
                                                'scores': U,
                                                'svals': D,
                                                'loadings': V,
                                                'rank': rank}

            ###################################
            # step 3.3: estimate noise matrix #
            ###################################

            if self.store_full and not issparse(X):
                E = X - (J + I)
            else:
                E = None
            block_specific[bn]['noise'] = E

        # save block specific estimates
        self.blocks = {}
        for bn in block_specific.keys():
            self.blocks[bn] = \
                BlockSpecificResults(joint=block_specific[bn]['joint'],
                                     individual=block_specific[bn]['individual'],
                                     noise=block_specific[bn]['noise'],
                                     CNS=joint_scores,
                                     block_name=bn,
                                     obs_names=obs_names,
                                     var_names=var_names[bn],
                                     m=self.centers_[bn],
                                     shape=blocks[bn].shape,
                                     zero_index_names=self.zero_index_names,
                                     init_signal_svd=init_signal_svd[bn],
                                     X=blocks[bn])

        return self

    @property
    def is_fit(self):
        if hasattr(self, 'blocks'):
            return True
        else:
            return False

    @property
    def block_names(self):
        if self.is_fit:
            return list(self.blocks.keys())
        else:
            return None

    def plot_joint_diagnostic(self, fontsize=20):
        """
        Plots joint rank threshold diagnostic plot
        """

        plot_joint_diagnostic(joint_svals=self.all_joint_svals_,
                              wedin_sv_samples=self.wedin_sv_samples_,
                              min_signal_rank=min(self.init_signal_ranks.values()),
                              random_sv_samples=self.random_sv_samples_,
                              wedin_percentile=self.wedin_percentile,
                              random_percentile=self.randdir_percentile,
                              fontsize=fontsize)

    def save(self, fpath, compress=9):
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        return load(fpath)

    def get_full_block_estimates(self):
        """

        Output
        ------
        full: dict of dict of np.arrays
        The joint, individual, and noise full estimates for each block.

        """
        full = {}
        for bn in self.block_names:
            full[bn] = {'joint': self.blocks[bn].joint.full_,
                        'individual': self.blocks[bn].individual.full_,
                        'noise': self.blocks[bn].noise_}

        return full

    def results_dict(self):
        """
        Returns all estimates as a dicts.

        """
        results = {}
        results['common'] = {'scores': self.common.scores_,
                             'svals': self.common.svals_,
                             'loadings': self.common.loadings_,
                             'rank': self.common.rank}

        for bn in self.block_names:
            joint = self.blocks[bn].joint
            indiv = self.blocks[bn].individual

            results[bn] = {'joint': {'scores': joint.scores_,
                                     'svals': joint.svals_,
                                     'loadings': joint.loadings_,
                                     'rank': joint.rank,
                                     'full': joint.full_},

                           'individual': {'scores': indiv.scores_,
                                          'svals': indiv.svals_,
                                          'loadings': indiv.loadings_,
                                          'rank': indiv.rank,
                                          'full': indiv.full_},

                           'noise': self.blocks[bn].noise_}

        return results

    def get_ranks(self):
        """
        Output
        ------
        joint_rank (int): the joint rank

        indiv_ranks (dict): the individual ranks.
        """
        if not self.is_fit:
            raise ValueError('Decomposition has not yet been computed')

        joint_rank = self.common.rank
        indiv_ranks = {bn: self.blocks[bn].individual.rank for bn in self.block_names}
        return joint_rank, indiv_ranks


def _dict_formatting(x):
    if hasattr(x, 'keys'):
        names = list(x.keys())
        assert len(set(names)) == len(names)
    else:
        names = list(range(len(x)))
    return {n: x[n] for n in names}


def arg_checker(blocks, init_signal_ranks, joint_rank, indiv_ranks,
                precomp_init_svd, center):
    """

    """
    # TODO: document
    # TODO: change assert to raise ValueError with informative message

    ##########
    # blocks #
    ##########

    blocks = _dict_formatting(blocks)
    block_names = list(blocks.keys())

    # check blocks have the same number of observations
    assert len(set(blocks[bn].shape[0] for bn in block_names)) == 1

    # get obs and variable names
    obs_names = list(range(list(blocks.values())[0].shape[0]))
    var_names = {}
    for bn in block_names:
        if type(blocks[bn]) == pd.DataFrame:
            obs_names = list(blocks[bn].index)
            var_names[bn] = list(blocks[bn].columns)
        else:
            var_names[bn] = list(range(blocks[bn].shape[1]))

    # format blocks
    # make sure blocks are either csr or np.array
    for bn in block_names:
        if issparse(blocks[bn]):  # TODO: allow for general linear operators
            raise NotImplementedError
            # blocks[bn] = csr_matrix(blocks[bn])
        else:
            blocks[bn] = np.array(blocks[bn])

    shapes = {bn: blocks[bn].shape for bn in block_names}

    ####################
    # precomp_init_svd #
    ####################
    if precomp_init_svd is None:
        precomp_init_svd = {bn: None for bn in block_names}
    precomp_init_svd = _dict_formatting(precomp_init_svd)
    assert set(precomp_init_svd.keys()) == set(block_names)
    for bn in block_names:
        udv = precomp_init_svd[bn]
        if udv is not None and not hasattr(udv, 'keys'):
            precomp_init_svd[bn] = {'scores': udv[0],
                                    'svals': udv[1],
                                    'loadings': udv[2]}

    # TODO: check either None or SVD provided
    # TODO: check correct SVD formatting
    # TODO: check SVD ranks are the same
    # TODO: check SVD rank is at least init_signal_ranks + 1

    #####################
    # init_signal_ranks #
    #####################
    if precomp_init_svd is None:
        precomp_init_svd = {bn: None for bn in block_names}
    init_signal_ranks = _dict_formatting(init_signal_ranks)
    assert set(init_signal_ranks.keys()) == set(block_names)

    # initial signal rank must be at least one lower than the shape of the block
    for bn in block_names:
        assert 1 <= init_signal_ranks[bn]
        assert init_signal_ranks[bn] <= min(blocks[bn].shape) - 1

    ##############
    # joint_rank #
    ##############
    if joint_rank is not None and joint_rank > sum(init_signal_ranks.values()):
        raise ValueError('joint_rank must be smaller than the sum of the initial signal ranks')

    ###############
    # indiv_ranks #
    ###############
    if indiv_ranks is None:
        indiv_ranks = {bn: None for bn in block_names}
    indiv_ranks = _dict_formatting(indiv_ranks)
    assert set(indiv_ranks.keys()) == set(block_names)

    for k in indiv_ranks.keys():
        assert indiv_ranks[k] is None or type(indiv_ranks[k]) in [int, float]
        # TODO: better check for numeric

    ##########
    # center #
    ##########
    if type(center) == bool:
        center = {bn: center for bn in block_names}
    center = _dict_formatting(center)

    return blocks, init_signal_ranks, indiv_ranks, precomp_init_svd, center,\
        obs_names, var_names, shapes


def indiv_space_for_sparse(X, joint_scores, joint_rank, signal_rank, sv_threshold):
    # compute a rank R1 SVD of I
    # if the estimated individual rank is less than R1 we are done
    # otherwise compute a rank R2 SVD of I
    # keep going until we find the individual rank
    # TODO: this could use lots of optimizing

    X_orthog = col_proj_orthog(X, joint_scores)

    # start with a low rank SVD
    max_rank = min(X.shape) - joint_rank  # saves computation
    current_rank = min(int(1.2 * signal_rank), max_rank)  # 1.2 is somewhat arbitrary
    U, D, V = svd_wrapper(X_orthog, current_rank)
    indiv_rank = sum(D > sv_threshold)

    if indiv_rank == current_rank:  # SVD rank is still too low
        found_indiv_rank = False
        for t in range(3):

            # current guess at an upper bound for the individual rank
            additional_rank = signal_rank
            current_rank = current_rank + additional_rank
            current_rank = min(current_rank, max_rank)

            # compute additional additional_rank SVD components

            # TODO: possibly use svds_additional to speed up calculation
            # U, D, V = svds_additional(I, scores, sv, loadings, additional_rank)
            U, D, V = svd_wrapper(X_orthog, current_rank)
            indiv_rank = sum(D > sv_threshold)

            # we are done if the individual rank estimate is less
            # than the current_rank or if the current_rank is equal to the maximal rank
            if (indiv_rank < current_rank) or (current_rank == max_rank):
                found_indiv_rank = True
                break

        if not found_indiv_rank:
            warnings.warn('individual rank estimate probably too low')

    return U[:, 0:indiv_rank], D[0:indiv_rank], V[:, 0:indiv_rank], indiv_rank


def reconsider_joint_components(blocks, sv_threshold,
                                joint_scores, joint_svals, joint_loadings,
                                joint_rank):
    """
    Checks the identifiability constraint on the joint singular values

    TODO: document
    """

    # check identifiability constraint
    to_keep = set(range(joint_rank))
    for bn in blocks.keys():
        for j in range(joint_rank):
            # This might be joint_sv
            score = np.dot(blocks[bn].T, joint_scores[:, j])
            sv = np.linalg.norm(score)

            # if sv is below the threshold for any data block remove j
            if sv < sv_threshold[bn]:
                # TODO: should probably keep track of this
                print('removing column ' + str(j))
                to_keep.remove(j)
                break

    # remove columns of joint_scores that don't satisfy the constraint
    joint_rank = len(to_keep)
    joint_scores = joint_scores[:, list(to_keep)]
    joint_loadings = joint_loadings[:, list(to_keep)]
    joint_svals = joint_svals[list(to_keep)]
    return joint_scores, joint_svals, joint_loadings, joint_rank


class BlockSpecificResults(object):
    """
    Contains the block specific results.

    Parameters
    ----------
    joint: dict
        The block specific joint PCA.

    individual: dict
        The block specific individual PCA.

    noise: array-like
        The noise matrix estimate.

    obs_names: None, array-like
        Observation names.

    var_names: None, array-like
        Variable names for this block.

    block_name: None, int, str
        Name of this block.

    m: None, array-like
        The vector used to column mean center this block.


    Attributes
    ----------
    joint: jive.PCA.PCA
        Block specific joint PCA.
        Has an extra attribute joint.full_ which contains the full block
        joint estimate.

    individual: jive.PCA.PCA
        Block specific individual PCA.
        Has an extra attribute individual.full_ which contains the full block
        joint estimate.


    noise: array-like
        The full noise block estimate.

    block_name:
        Name of this block.

    """
    def __init__(self, joint, individual, noise, CNS,  # X,
                 obs_names=None, var_names=None, block_name=None,
                 m=None, shape=None, zero_index_names=True,
                 init_signal_svd=None, X=None):

        self.joint = PCA.from_precomputed(n_components=joint['rank'],
                                          scores=joint['scores'],
                                          loadings=joint['loadings'],
                                          svals=joint['svals'],
                                          obs_names=obs_names,
                                          var_names=var_names,
                                          m=m, shape=shape)

        if joint['rank'] != 0:
            base = 'joint'
            if block_name is not None:
                base = '{}_{}'.format(block_name, base)
            self.joint.set_comp_names(base=base, zero_index=zero_index_names)

        if joint['full'] is not None:
            self.joint.full_ = pd.DataFrame(joint['full'],
                                            index=obs_names, columns=var_names)
        else:
            self.joint.full_ = None

        self.individual = PCA.from_precomputed(n_components=individual['rank'],
                                               scores=individual['scores'],
                                               loadings=individual['loadings'],
                                               svals=individual['svals'],
                                               obs_names=obs_names,
                                               var_names=var_names,
                                               m=m, shape=shape)
        if individual['rank'] != 0:
            base = 'indiv'
            if block_name is not None:
                base = '{}_{}'.format(block_name, base)
            self.individual.set_comp_names(base=base,
                                           zero_index=zero_index_names)

        if individual['full'] is not None:
            self.individual.full_ = pd.DataFrame(individual['full'],
                                                 index=obs_names,
                                                 columns=var_names)
        else:
            self.individual.full_ = None

        if noise is not None:
            self.noise_ = pd.DataFrame(noise, index=obs_names,
                                       columns=var_names)
        else:
            self.noise_ = None

        self.block_name = block_name

        # compute common normalized loadings
        # U, D, V = self.joint.get_UDV()

        U, D, V = init_signal_svd['scores'], init_signal_svd['svals'], \
            init_signal_svd['loadings']
        common_loadigs = V.dot(np.multiply(U, 1.0 / D).T.dot(CNS))
        # common_loadigs = V.dot(np.multiply(U, D).T.dot(CNS))
        # col_norms = np.linalg.norm(common_loadigs, axis=0)
        # common_loadigs *= (1.0 / col_norms)

        base = 'common'
        if block_name is None:
            base = '{}_{}'.format(block_name, base)
        comp_names = get_comp_names(base=base, num=CNS.shape[1],
                                    zero_index=zero_index_names)
        self.common_loadings_ = pd.DataFrame(common_loadigs,
                                             index=var_names,
                                             columns=comp_names)

        # TODO: delete
        # # regression on J
        # U, D, V = joint['scores'], joint['svals'], joint['loadings']
        # common_loadings_reg_J = \
        #     V.dot(np.multiply(U, 1.0 / D).T.dot(CNS))
        # self.common_loadings_reg_J = pd.DataFrame(common_loadings_reg_J,
        #                                           index=var_names,
        #                                           columns=comp_names)

        # regression on X
        common_loadings_reg_X = []
        for j in range(CNS.shape[1]):
            lm = LinearRegression().fit(X, CNS[:, j])
            common_loadings_reg_X.append(lm.coef_)
        common_loadings_reg_X = np.array(common_loadings_reg_X).T
        self.common_loadings_reg_X = pd.DataFrame(common_loadings_reg_X,
                                                  index=var_names,
                                                  columns=comp_names)

    def common_loadings(self, np=False):
        # TODO: should we keep this function to match the PCA object?
        if np:
            return self.common_loadings_.values
        else:
            return self.common_loadings_

    def plot_common_loading(self, comp, abs_sorted=True, show_var_names=True,
                            significant_vars=None, show_top=None, title=True):
        """
        Plots the values for each feature of a single loading component.

        Parameters
        ----------
        comp: int
            Which PCA component.

        abs_sorted: bool
            Whether or not to sort components by their absolute values.

        significant_vars: {None, array-like}, shape (n_featurse, )
            Indicated which features are significant in this component.

        show_top: {None, int}
            Will only display this number of top loadings components when
            sorting by absolute value.

        title: {str, bool}
            Plot title. User can provide their own otherwise will
            use default title.
        """
        plot_loading(v=self.common_loadings_.iloc[:, comp],
                     abs_sorted=abs_sorted, show_var_names=show_var_names,
                     significant_vars=significant_vars, show_top=show_top)

        if type(title) == str:
            plt.title(title)
        elif title:
            plt.title('common loadings comp {}'.format(comp))

    def __repr__(self):
        return 'Block: {}, individual rank: {}, joint rank: {}'.format(self.block_name, self.individual.rank, self.joint.rank)
