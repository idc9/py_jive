import numpy as np
import pandas as pd
from joblib import load, dump
import matplotlib.pyplot as plt
from numpy.linalg import norm
from scipy.sparse import issparse

from jive.utils import svd_wrapper, centering

from jive.viz.viz import plot_loading, plot_scores_hist, interactive_slice
from jive.viz.singular_values import scree_plot, plot_var_expl_prop, \
    plot_var_expl_cum
from jive.viz.pca_scores_plot import scores_plot


# from bokeh.io import output_notebook# , push_notebook, show

# TODOs
# - finish documentation
# - make documentation follow sklearn conventions more closely.
# - implement methods for automatic PCA rank selection
# - interface with JackStraw
class PCA(object):
    """
    Computes the Principal Components Analysis (PCA) of a data matrix
    X (n_samples x n_features).

    Parameters
    ----------
    n_components: None, int
        rank of the decomposition. If None, will compute full PCA.

    center: str, None
        How to center the columns of X. If None, will not center the
        columns (i.e. just computes the SVD).


    Attributes
    ----------
    scores_: pd.DataFrame, shape (n_samples, n_components)
        The orthonormal matrix of (normalized) scores.

    loadings_: pd.DataFrame, shape (n_features, n_components)
        The orthonormal matrix of loadings.

    svals_: pd.Series, shape (n_components, )
        The singular values.

    m_: np.array, shape (n_features, )
        The vector used to center the data.

    frob_norm_: float
        The Frobenius norm of the training data matrix X.

    shape_: tuple length 2
        The shape of the original data matrix.
    """
    def __init__(self, n_components=None, center='mean'):
        self.n_components = n_components
        self.center = center

    def get_params(self):
        return {'n_components': self.n_components,
                'center': self.center}

    def __repr__(self):
        if not hasattr(self, 'scores_'):
            return 'PCA object, nothing has been computed yet'
        else:
            return 'Rank {} PCA of a {} matrix'.format(self.n_components, self.shape_)

    def fit(self, X):
        """
        Computes the PCA decomposition of X.

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Fit PCA with data matrix X. If X is a pd.DataFrame, the observation
            and feature names will be extracted from its index/columns.
            Note X can be either dense or sparse.

        """
        self.shape_, obs_names, var_names, self.n_components, \
            = _arg_checker(X, self.n_components)

        # possibly mean center X
        X, self.m_ = centering(X, self.center)

        # compute SVD
        U, D, V = svd_wrapper(X, self.n_components)

        # compute variance explained
        if self.n_components == min(X.shape):
            self.frob_norm_ = np.sqrt(sum(D ** 2))
        else:
            self.frob_norm_ = _safe_frob_norm(X)
        self.var_expl_prop_ = D ** 2 / self.frob_norm_ ** 2
        self.var_expl_cum_ = np.cumsum(self.var_expl_prop_)

        if self.n_components is None:
            self.n_components = self.scores_.shape[1]

        self.scores_, self.svals_, self.loadings_ = \
            svd2pd(U, D, V, obs_names=obs_names, var_names=var_names)

        return self

    @classmethod
    def from_precomputed(cls, n_components=None, center=None,
                         scores=None, loadings=None, svals=None,
                         obs_names=None, var_names=None, comp_names=None,
                         m=None, frob_norm=None, var_expl_prop=None,
                         shape=None):

        """
        Loads the PCA object from a precomputed PCA decomposition.
        """

        x = cls()
        if n_components is None and scores is not None:
            n_components = scores.shape[1]
        x.n_components = n_components

        if shape is not None:
            shape = shape
        else:
            shape = [None, None]
            if scores is not None:
                shape[0] = scores.shape[0]
            if loadings is not None:
                shape[1] = loadings.shape[0]
        x.shape_ = shape

        if scores is not None and type(scores) != pd.DataFrame:
            if obs_names is None:
                obs_names = _default_obs_names(scores.shape[0])
            if comp_names is None:
                comp_names = _default_comp_names(scores.shape[1])
            scores = pd.DataFrame(scores, index=obs_names,
                                  columns=comp_names)

        if svals is not None and type(svals) != pd.Series:
            if comp_names is None:
                comp_names = _default_comp_names(loadings.shape[1])
            svals = pd.Series(svals, index=comp_names)

        if loadings is not None and type(loadings) != pd.DataFrame:
            if var_names is None:
                var_names = _default_var_names(loadings.shape[0])
            if comp_names is None:
                comp_names = _default_comp_names(loadings.shape[1])
            loadings = pd.DataFrame(loadings, index=var_names,
                                    columns=comp_names)
        x.scores_ = scores
        x.loadings_ = loadings
        x.svals_ = svals

        x.center = center
        x.m_ = m

        x.frob_norm_ = frob_norm
        x.var_expl_prop_ = var_expl_prop
        if var_expl_prop is not None:
            x.var_expl_cum_ = np.cumsum(var_expl_prop)
        else:
            x.var_expl_cum_ = None

        return x

    def save(self, fpath, compress=9):
        """
        Saves the PCA object to disk using sklearn.externals.joblib

        Parameters
        ----------
        fpath: str
            Path to saved file.

        compress: int
            Level of compression. See documentation of
            sklearn.externals.joblib.dump
        """
        dump(self, fpath, compress=compress)

    @classmethod
    def load(cls, fpath):
        """
        Loads a PCA object from disk.

        Parameters
        ----------
        fpath: (str)
            Path to saved file.

        Output
        ------
        jive.PCA.PCA
        """
        return load(fpath)

    @property
    def rank(self):  # synonym of n_components
        return self.n_components

    def obs_names(self):
        """
        Returns the observation names.
        """
        return np.array(self.scores_.index)

    def comp_names(self):
        """
        Returns the component names.
        """
        return np.array(self.scores_.columns)

    def var_names(self):
        """
        Returns the variable names.
        """
        return np.array(self.loadings_.index)

    def set_comp_names(self, comp_names=None, base=None, zero_index=True):
        """
        Resets the component names.
        """
        if comp_names is None:
            comp_names = get_comp_names(base=base,
                                        num=len(self.scores_.columns),
                                        zero_index=zero_index)

        self.scores_.columns = comp_names
        self.svals_.index = comp_names
        self.loadings_.columns = comp_names
        return self

    def scores(self, norm=True, np=False):
        """

        Returns the scores.

        Parameters
        ----------
        norm: bool
            If true, returns normalized scores. Otherwise, returns unnormalized
            scores.

        np: bool
            If true, returns scores as a numpy array. Otherwise, returns pandas.

        """
        if norm:  # normalized scores
            if np:
                return self.scores_.values
            else:
                return self.scores_
        else:

            unnorm_scores = _unnorm_scores(self.scores_, self.svals_)
            if np:
                return unnorm_scores
            else:
                return pd.DataFrame(unnorm_scores,
                                    index=self.scores_.index,
                                    columns=self.scores_.columns)

    def loadings(self, np=False):
        if np:
            return self.loadings_.values
        else:
            return self.loadings_

    def svals(self, np=False):
        if np:
            return self.svals_.values
        else:
            return self.svals_

    def get_UDV(self):
        """
        Returns the Singular Value Decomposition of (possibly centered) X.

        Output
        ------
        U, D, V

        U: np.array (n_samples, n_components)
            scores (left singular values)

        D: np.array (n_components, )
            singular values

        V: np.array (n_features, n_components)
            loadings matrix (right singular values)
        """
        return self.scores_.values, self.svals_.values, self.loadings_.values

    def predict_scores(self, X):
        """
        Projects a new data matrix Y onto the loadings and returns the
        coordinates (scores) in the PCA subspace.

        Parameters
        ----------
        X: array-like, shape (n_new_samples, n_features)
        """
        s = np.dot(X, self.loadings_)
        if self.m_ is not None:
            s -= np.dot(self.m_, self.loadings_)
        return s

    def predict_reconstruction(self, X=None):
        """
        Reconstructs the data in the original spaces (R^n_features). I.e projects
        each data point onto the rank n_components PCA affine subspace
        which sits in the original n_features dimensional space.


        Parameters
        ----------
        Y: None, array-like shape(n_new_samples, n_features)
            Projects data onto PCA subspace which live in the original
            space (R^n_features). If None, will use return the reconstruction
            of the training ddata.

        """
        # TODO: should we make a separate predict_train_reconstruction function?
        if X is None:
            proj = _unnorm_scores(self.scores_.values, self.svals_)
        else:
            proj = self.predict_scores(X)

        return pca_reconstruct(proj, V=self.loadings_, m=self.m_)

    def reconstruction_error(self, X):
        """
        Computes the mean squared reconstruction error i.e.
        ||X_hat - X||_F^2 / (X.shape[0] * X.shape[1])

        Parameters
        ----------
        X array-like, shape (n_new_samples, n_features)
        """
        X_hat = self.predict_reconstruction(X)
        sq_diffs = (X_hat - np.array(X)).reshape(-1) ** 2

        return np.mean(sq_diffs)

    def score(self, X, y=None):
        """
        Returns the mean squared reconstruction error from the samples.
        Makes this class sklearn compatible.
        """
        # TODO: confusing notation: score and scores, what should we do about this?
        return self.reconstruction_error(X)

    def plot_loading(self, comp, abs_sorted=True, show_var_names=True,
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
        plot_loading(v=self.loadings().iloc[:, comp],
                     abs_sorted=abs_sorted, show_var_names=show_var_names,
                     significant_vars=significant_vars, show_top=show_top)

        if type(title) == str:
            plt.title(title)
        elif title:
            plt.title('loadings comp {}'.format(comp))

    def plot_scores_hist(self, comp, norm=True, **kwargs):
        """
        Plots jitter-histogram of one scores component.

        Parameters
        ----------

        comp: int
            Which component.

        norm: bool
            Whether to use normalized scores.

        **kwargs:
            keyword arguments for plt.hist
        """
        plot_scores_hist(self.scores(norm=norm).iloc[:, comp], comp=comp, **kwargs)

    def plot_scree(self, log=False, diff=False):
        """
        Makes a scree plot of the singular values.

        Parameters
        ----------
        log: bool
            Take log base 10 of singular values.

        diff: bool
            Plot difference of successive singular values.
        """
        scree_plot(self.svals_.values, log=log, diff=diff)

    def plot_var_expl_prop(self):
        """
        Plots the proportion of variance explained for each component.
        """
        plot_var_expl_prop(self.var_expl_prop_)

    def plot_var_expl_cum(self):
        """
        Plots the cumulative variance explained.
        """
        plot_var_expl_cum(self.var_expl_cum_)

    def plot_scores(self, norm=True,
                    start=0, n_components=3, classes=None, class_name=None,
                    dist_kws={}, scatter_kws={}):

        """
        Scores plot. See documentation of jive.viz.pca_scores_plot.scores_plot.

        Parameters
        ----------
        norm: bool
            Plot normalized scores.
        """

        scores_plot(self.scores(norm=norm),
                    start=start,
                    ncomps=n_components,
                    classes=classes,
                    class_name=class_name,
                    dist_kws=dist_kws,
                    scatter_kws=scatter_kws)

    def plot_scores_vs(self, comp, y, norm=True, ylabel=''):
        """
        Scatter plot of one scores component vs. a continuous variable.

        Parameters
        ----------
        comp: int
            Which component.

        y: (array-like), shape (n_samples, )
            Variable to plot against.

        norm: bool
            Use normalized scores.

        ylabel: str
            Name of the variable.
        """

        s = self.scores(norm=norm).iloc[:, comp]

        corr = np.corrcoef(s, y)[0, 1]
        plt.scatter(s, y)
        plt.xlabel('comp {} scores'.format(comp))
        plt.ylabel(ylabel)
        plt.title('correlation: {:1.4f}'.format(corr))

    def scores_corr_vs(self, y):
        """
        Computes the correlation between each PCA component and a continuous
        variable.
        """
        return np.array([np.corrcoef(self.scores().iloc[:, i], y)[0, 1]
                         for i in range(self.n_components)])

    def plot_interactive_scores_slice(self, comp1, comp2, norm=True, classes=None):
        """
        Makes an interactive scatter plot of the scores from two components.
        The user can drag the mouse to select a set of observations then
        get their index values as a pandas data frame. See documentation.

        Parameters
        ----------
        comp1, comp2: int
            The component indices.

        norm: bool
            Use normalized scores.

        cats: {list, None}, shape (n_samples, )
            Categories to color points by.


        Example usage
        -------------
        import numpy as np
        from jive.PCA import PCA

        pca = PCA().fit(np.random.normal(size=(100, 20)))
        model, saved_selected = pca.plot_interactive_scores_slice(0, 1)

        # user selects some points using Lasso Select tool

        model.to_df() the contains a pd.DataFrame listing the selected points
        """
        scores = self.scores(norm=norm)
        return interactive_slice(x=scores.iloc[:, comp1],
                                 y=scores.iloc[:, comp2],
                                 cats=classes,
                                 obs_names=self.obs_names(),
                                 xlab='component {}'.format(comp1),
                                 ylab='component {}'.format(comp2))


def _arg_checker(X, n_components):

    if n_components is None:
        n_components = min(X.shape)

    assert n_components >= 1 and n_components <= min(X.shape)

    # extract data from X
    shape = X.shape

    # extract observation/variable names
    if type(X) == pd.DataFrame:
        obs_names = np.array(X.index)
        var_names = np.array(X.columns)
    else:
        obs_names = None
        var_names = None

    return shape, obs_names, var_names, n_components


def _default_obs_names(n_samples):
    return [i for i in range(n_samples)]


def _default_var_names(n_features):
    return ['feat_{}'.format(i) for i in range(n_features)]


def _default_comp_names(n_components):
    return ['comp_{}'.format(i) for i in range(n_components)]


def svd2pd(U, D, V, obs_names=None, var_names=None, comp_names=None):
    """
    Converts SVD output from numpy arrays to pandas.
    """
    if obs_names is None:
        obs_names = _default_obs_names(U.shape[0])

    if var_names is None:
        var_names = _default_var_names(V.shape[0])

    if comp_names is None:
        comp_names = _default_comp_names(U.shape[1])

    U = pd.DataFrame(U, index=obs_names, columns=comp_names)
    D = pd.Series(D, index=comp_names)
    V = pd.DataFrame(V, index=var_names, columns=comp_names)

    return U, D, V


def _unnorm_scores(U, D):
    """
    Returns the unnormalized scores.

    Parameters
    ----------
    U: array-like, shape (n_samples, n_components)
        Normalized scores.

    D: array-like, shape (n_components)
        Singular values.

    """
    _U = np.array(U)
    if _U.ndim == 1:  # if U is a vector, then return as a vector
        is_vec = True
    else:
        is_vec = False

    if is_vec or _U.shape[1] == 1:
        UD = _U.reshape(1, -1) * np.array(D)
    else:
        UD = _U * np.array(D)

    return UD


def pca_reconstruct(proj, V, m=None):
    """
    Let the rank K pca of X be given by X ~= U D V^T. X in R^n x d
    where n = number of observations and d = number of variables.

    For a given set of scores returns the predicted reconstruction of X.
    For example, if u_i is the ith row of U (the scores for the
    ith observation) then this returns V D u_i + m.

    Parameters
    ---------
    proj: the projections of the data onto the PCA subspace i.e.
        for the training data proj = UD

    V: the loadings (nd.array of dimension d x K)

    m: the mean of the data (vector in R^d)
    """

    R = np.dot(proj, V.T)
    if m is not None:
        R += m

    if np.array(proj).ndim == 1:  # if proj is a vector, then return as a vector
        return R.reshape(-1)
    else:
        return R


def _safe_frob_norm(X):
    """
    Calculates the Frobenius norm of X whether X is dense or sparse.

    Currently, neither scipy.linalg.norm nor numpy.linalg.norm work for
    sparse matrices.
    """
    if issparse(X):
        return np.sqrt(sum(X.data ** 2))
    else:
        return norm(np.array(X), ord='fro')


def get_comp_names(base, num, zero_index=True):
    if zero_index:
        start = 0
        stop = num
    else:
        start = 1
        stop = num + 1

    return ['{}_{}'.format(base, i) for i in range(start, stop)]
