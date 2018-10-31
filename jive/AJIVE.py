import numpy as np
import warnings
from scipy.sparse import issparse
from copy import deepcopy
from sklearn.externals.joblib import load, dump
import pandas as pd

from jive.utils import svd_wrapper, centering
from jive.lazymatpy.templates.matrix_transformations import col_proj, col_proj_orthog
from jive.wedin_bound import get_wedin_samples
from jive.random_direction import sample_randdir
from jive.viz.diagnostic_plot import plot_joint_diagnostic

from jive.PCA import PCA

# TODOs
# - finish documentation
# - add test cases
# - make documentation follow sklearn conventions more closely


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

    wedin_percentile: int (default=95)
        Percentile for wedin bound cutoff for estimating joint rank.

    wedin_n_samples: int (default=1000)
        Number of wedin bound samples to draw.

    precomp_wedin_samples {None, dict of array-like, list of array-like}
        Precomputed Wedin samples for each data block.

    randdir_percentile: int (default=5)
        Percentile for random direction bound cutoff
        for estimating joint rank.

    randdir_n_samples: int (default=1000)
        Number of random directions samples to draw.

    precomp_randdir_samples {None,  array-like}
        Precomputed random direction samples.

    Attributes
    ----------

    common: jive.PCA.PCA
        The common joint space.

    blocks: dict of BlockSpecificResults
        The block specific results.


    centers_:


    sv_threshold_:


    all_joint_svs_:


    random_sv_samples_:

    wedin_samples_:

    wedin_cutoff_:

    rand_cutoff_:

    svsq_cutoff_:

    joint_rank_wedin_est_:

    joint_rank:

    indiv_ranks:

    """

    def __init__(self,
                 init_signal_ranks,
                 joint_rank=None, indiv_ranks=None,
                 center='mean',
                 reconsider_joint_components=True,
                 wedin_percentile=95, wedin_n_samples=1000,
                 precomp_wedin_samples=None,
                 randdir_percentile=5, randdir_n_samples=1000,
                 precomp_randdir_samples=None,
                 store_full=True):

        self.init_signal_ranks = init_signal_ranks
        self.joint_rank = joint_rank
        self.indiv_ranks = indiv_ranks

        self.center = center

        self.wedin_percentile = wedin_percentile
        self.wedin_n_samples = wedin_n_samples
        self.wedin_samples_ = precomp_wedin_samples

        self.randdir_percentile = randdir_percentile
        self.randdir_n_samples = randdir_n_samples
        self.random_sv_samples_ = precomp_randdir_samples

        self.reconsider_joint_components = reconsider_joint_components

        self.store_full = store_full

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
        Parameters
        ----------
        blocks (list, dict): the data matrices.

        precomp_init_svd (list, dict, None): precomputed initial SVD.
        """
        blocks, self.init_signal_ranks, self.indiv_ranks, precomp_init_svd,\
            obs_names, var_names = arg_checker(blocks,
                                               self.init_signal_ranks,
                                               self.joint_rank,
                                               self.indiv_ranks,
                                               precomp_init_svd)
        block_names = list(blocks.keys())
        num_obs = list(blocks.values())[0].shape[0]

        # center blocks
        self.centers_ = {}
        for bn in block_names:
            blocks[bn], self.centers_[bn] = centering(blocks[bn],
                                                      method=self.center)

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
                D = precomp_init_svd[bn]['svs']
                V = precomp_init_svd[bn]['loadings']

            # The SV threshold is halfway between the init_signal_ranks[bn]th
            # and init_signal_ranks[bn] + 1 st singular value. Recall that
            # python is zero indexed.
            self.sv_threshold_[bn] = (D[self.init_signal_ranks[bn] - 1] \
                                      + D[self.init_signal_ranks[bn]])/2

            init_signal_svd[bn] = {'scores': U[:, 0:self.init_signal_ranks[bn]],
                                   'svs': D[0:self.init_signal_ranks[bn]],
                                   'loadings': V[:, 0:self.init_signal_ranks[bn]]}

        ##################################
        # step 2: joint space estimation #
        ##################################
        # this step estimates the joint rank and computes the common
        # joint space basis

        # SVD of joint signal matrix
        joint_scores_matrix = np.bmat([init_signal_svd[bn]['scores'] for bn in block_names])
        joint_scores, joint_svs, joint_loadings = svd_wrapper(joint_scores_matrix)
        self.all_joint_svs_ = deepcopy(joint_svs)

        # estimate joint rank using wedin bound and random direction if a
        # joint rank estimate has not already been provided
        # TODO: maybe make this into an object or function
        if self.joint_rank is None:

            # if the random sv samples are not already provided compute them
            if self.random_sv_samples_ is None:
                self.random_sv_samples_ = \
                    sample_randdir(num_obs,
                                   signal_ranks=list(self.init_signal_ranks.values()),
                                   num_samples=self.randdir_n_samples)

            # if the wedin samples are not already provided compute them
            if self.wedin_samples_ is None:
                self.wedin_samples_ = {}
                for bn in block_names:
                    self.wedin_samples_[bn] = \
                        get_wedin_samples(X=blocks[bn],
                                          U=init_signal_svd[bn]['scores'],
                                          D=init_signal_svd[bn]['svs'],
                                          V=init_signal_svd[bn]['loadings'],
                                          rank=self.init_signal_ranks[bn],
                                          num_samples=self.wedin_n_samples)

            self.wedin_sv_samples_ = len(blocks) - \
                np.array([sum(self.wedin_samples_[bn][i] ** 2 for bn in block_names)
                          for i in range(self.wedin_n_samples)])

            # given the wedin and random bound samples, compute the joint rank
            # SV cutoff
            self.wedin_cutoff_ = np.percentile(self.wedin_sv_samples_,
                                               self.wedin_percentile)
            self.rand_cutoff_ = np.percentile(self.random_sv_samples_,
                                              self.randdir_percentile)
            self.svsq_cutoff_ = max(self.wedin_cutoff_, self.rand_cutoff_)
            self.joint_rank_wedin_est_ = sum(joint_svs ** 2 > self.svsq_cutoff_)
            self.joint_rank = deepcopy(self.joint_rank_wedin_est_)

        # check identifiability constraint and possibly remove some
        # joint components
        if self.reconsider_joint_components:
            joint_scores, joint_svs, joint_loadings, self.joint_rank = \
                reconsider_joint_components(blocks, self.sv_threshold_,
                                            joint_scores, joint_svs, joint_loadings,
                                            self.joint_rank)

        # TODO: include center?
        # The common joint space has now been estimated
        self.common = PCA.from_precomputed(scores=joint_scores[:, 0:self.joint_rank],
                                           svals=joint_svs[0:self.joint_rank],
                                           loadings=joint_loadings[:, 0:self.joint_rank],
                                           obs_names=obs_names)

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

            if issparse(X):  # lazy evaluation for sparse matrices
                J = col_proj(X, joint_scores)
                U, D, V = svd_wrapper(J, self.joint_rank)
                J = np.array([])  # kill J matrix to save memory

            else:
                J = np.array(np.dot(joint_scores, np.dot(joint_scores.T, X)))
                U, D, V = svd_wrapper(J, self.joint_rank)
                if not self.store_full:
                    J = np.array([])  # kill J matrix to save memory

            block_specific[bn]['joint'] = {'full': J,
                                           'scores': U,
                                           'svs': D,
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
                I = np.array([])

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
                    U = U[:, 0:rank]
                    D = D[0:rank]
                    V = V[:, 0:rank]
                    self.indiv_ranks[bn] = rank

                else:  # indiv_rank has been provided by the user
                    rank = self.indiv_ranks[bn]
                    U, D, V = svd_wrapper(X_orthog, rank)

                if self.store_full:
                    I = np.array(np.dot(U, np.dot(np.diag(D), V.T)))
                else:
                    I = np.array([])  # Kill I matrix to save memory

            block_specific[bn]['individual'] = {'full': I,
                                                'scores': U,
                                                'svs': D,
                                                'loadings': V,
                                                'rank': rank}

            ###################################
            # step 3.3: estimate noise matrix #
            ###################################

            if self.store_full and not issparse(X):
                E = X - (J + I)
            else:
                E = np.array([])
            block_specific[bn]['noise'] = E

        # save block specific estimates
        self.blocks = {}
        for bn in block_specific.keys():
            self.blocks[bn] = BlockSpecificResults(joint=block_specific[bn]['joint'],
                                                   individual=block_specific[bn]['individual'],
                                                   noise=block_specific[bn]['noise'],
                                                   block_name=bn,
                                                   obs_names=obs_names,
                                                   var_names=var_names[bn],
                                                   m=self.centers_[bn])

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

    def plot_joint_diagnostic(self):
        """
        Plots joint rank threshold diagnostic plot
        """
        if not self.is_fit:
            raise ValueError('Decomposition has not yet been computed')

        plot_joint_diagnostic(joint_svs=self.all_joint_svs_,
                              wedin_sv_samples=self.wedin_sv_samples_,
                              random_sv_samples=self.random_sv_samples_,
                              wedin_percentile=self.wedin_percentile,
                              random_percentile=self.randdir_percentile)

    def save(self, fpath, compress=9):
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        return load(fpath)

    def get_full_block_estimates(self):
        full = {}
        for bn in self.block_names:
            full[bn] = {'joint': self.blocks[bn].joint.full_,
                        'individual': self.blocks[bn].individual.full_,
                        'noise': self.blocks[bn].noise_}

        return full

    def results_dict(self):
        results = {}
        results['common'] = {'scores': self.common.scores_,
                             'svs': self.common.svs_,
                             'loadings': self.common.loadings_,
                             'rank': self.common.rank}

        for bn in self.block_names:
            joint = self.blocks[bn].joint
            indiv = self.blocks[bn].individual

            results[bn] = {'joint': {'scores': joint.scores_,
                                     'svs': joint.svs_,
                                     'loadings': joint.loadings_,
                                     'rank': joint.rank,
                                     'full': joint.full_},

                           'individual': {'scores': indiv.scores_,
                                          'svs': indiv.svs_,
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
        indiv_ranks = {bn: self.blocks[bn].rank for bn in self.block_names}
        return joint_rank, indiv_ranks


# def _check_block_formatting(x, blocks):
#     assert type(x) == type(blocks)
#     assert len(x) == len(x)

#     if type(x) == dict:
#         assert set(x.keys()) == set(blocks.keys())

#     if type(x) == list:
#         x = {i: x[i] for i in range(len(x))}

#     return x


def _dict_formatting(x):
    if hasattr(x, 'keys'):
        names = list(x.keys())
        assert len(set(names)) == len(names)
    else:
        names = list(range(len(x)))
    return {n: x[n] for n in names}


def arg_checker(blocks, init_signal_ranks, joint_rank, indiv_ranks, precomp_init_svd):
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
    obs_names = list(range(list(blocks.values())[0].shape[1]))
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

    ####################
    # precomp_init_svd #
    ####################
    if precomp_init_svd is None:
        precomp_init_svd = {bn: None for bn in block_names}
    precomp_init_svd = _dict_formatting(precomp_init_svd)
    assert set(precomp_init_svd.keys()) == set(block_names)

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

    return blocks, init_signal_ranks, indiv_ranks, precomp_init_svd, obs_names, var_names


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
                                joint_scores, joint_svs, joint_loadings,
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
    joint_svs = joint_svs[list(to_keep)]
    return joint_scores, joint_svs, joint_loadings, joint_rank


class BlockSpecificResults():
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
    def __init__(self, joint, individual, noise,
                 obs_names=None, var_names=None, block_name=None,
                 m=None):
        self.joint = PCA.from_precomputed(n_components=joint['rank'],
                                          scores=joint['scores'],
                                          loadings=joint['loadings'],
                                          svals=joint['svs'],
                                          obs_names=obs_names,
                                          var_names=var_names,
                                          m=m)
        self.joint.full_ = joint['full']

        self.individual = PCA.from_precomputed(n_components=individual['rank'],
                                               scores=individual['scores'],
                                               loadings=individual['loadings'],
                                               svals=individual['svs'],
                                               obs_names=obs_names,
                                               var_names=var_names,
                                               m=m)
        self.individual.full_ = individual['full']

        self.noise_ = noise

        self.block_name = block_name

    def __repr__(self):
        return 'Block: {}, individual rank: {}, joint rank: {}'.format(self.block_name, self.individual.rank, self.joint.rank)