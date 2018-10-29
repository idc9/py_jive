import numpy as np
import pandas as pd
from sklearn.externals.joblib import load, dump
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
# - add testing
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
        how to center the columns of X. If None, will not center the
        columns (i.e. just computes the SVD).


    Attributes
    ----------
    scores_:

    loadings_:

    svals_:

    m_:

    frob_norm_

    obs_names_:

    var_names:

    comp_names_:

    shape_:
    """
    def __init__(self, n_components=None, center='mean'):
        self.n_components = n_components
        self.center = center

    def get_params(self):
        return {'n_components': self.n_components,
                'center': self.center}

    @classmethod
    def from_precomputed(cls, n_components=None, center=None,
                         scores=None, loadings=None, svals=None,
                         obs_names=None, var_names=None, m=None,
                         frob_norm=None, var_expl_prop=None):

        """
        Loads the PCA object from a precomputed PCA decomposition.
        """

        x = cls()
        if n_components is None and scores is not None:
            n_components = scores.shape[1]
        x.n_components = n_components

        shape = [None, None]
        if scores is not None:
            shape[0] = scores.shape[0]
        if loadings is not None:
            shape[1] = loadings.shape[0]
        x.shape_ = shape

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

        x.obs_names_ = obs_names
        x.var_names_ = var_names
        x.comp_names_ = ['comp_{}'.format(i) for i in range(x.n_components)]

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

    def __repr__(self):
        if not hasattr(self, 'scores_'):
            return 'PCA object, nothing has been computed yet'
        else:
            return 'Rank {} PCA of a {} matrix'.format(self.n_components, self.shape_)

    def fit(self, X):
        """

        Parameters
        ----------
        X: {array-like, sparse matrix}, shape (n_samples, n_features)
            Fit PCA with data matrix X. If X is a pd.DataFrame, obs_names_
            will be set to X.index and var_names_ will be set to X.columns.
            Note X can be either dense or sparse.

        """
        self.shape_, self.obs_names_, self.var_names_ = _arg_checker(X, self.n_components)

        # possibly mean center X
        X, self.m_ = centering(X, self.center)

        # compute SVD
        self.scores_, self.svals_, self.loadings_ = svd_wrapper(X, self.n_components)

        # compute variance explained
        if self.n_components == min(X.shape):
            self.frob_norm_ = np.sqrt(sum(self.svals_ ** 2))
        else:
            self.frob_norm_ = _safe_frob_norm(X)
        self.var_expl_prop_ = self.svals_ ** 2 / self.frob_norm_ ** 2
        self.var_expl_cum_ = np.cumsum(self.var_expl_prop_)

        if self.n_components is None:
            self.n_components = self.scores_.shape[1]

        self.comp_names_ = ['comp_{}'.format(i) for i in range(self.n_components)]
        return self

    @property
    def rank(self):  # synonym of n_components
        return self.n_components

    def scores(self):
        """
        Returns the (normalized) scores matrix, U in R^{n x K},
        as a pd.DataFrame indexed by var_names.
        """
        return pd.DataFrame(self.scores_, index=self.obs_names_,
                            columns=self.comp_names_)

    @property
    def unnorm_scores_(self):
        return np.dot(self.scores_, np.diag(self.svals_))

    def unnorm_scores(self):
        """
        Returns the unnormalized scores matrix, UD in R^{n x K},
        as a pd.DataFrame indexed by var_names.
        """
        return pd.DataFrame(self.unnorm_scores_,
                            index=self.obs_names_,
                            columns=self.comp_names_)

    def loadings(self):
        """
        Returns the loadings matrix, V in R^{d x K},
        as a pd.DataFrame indexed by var_names.
        """
        return pd.DataFrame(self.loadings_, index=self.var_names_,
                            columns=self.comp_names_)

    def predict_scores(self, Y):
        """
        Projects a new data matrix Y onto the loadings
        """
        if self.center:
            Y = Y - self.m_
        return np.dot(Y, self.loadings_)

    def reconstruct_from_scores(self, scores):
        return pca_reconstruct(u=scores, D=self.svals_, V=self.loadings_,
                               m=self.m_)

    def predict_reconstruction(self, y):
        """
        Parameters
        ----------
        y {int, array-like}:
            If y is an int, reconstructs the y'th observation of the
            training data. If y is vector, then computes the pca
            reconstruction of y.

        """
        if type(y) == int:
            assert (0 <= y) and (y <= self.shape_[0])
            u = self.scores_[y, :]

        else:
            u = np.dot(self.loadings_, y)

        return self.reconstruct_from_scores(scores=u)

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
        if norm:
            scores = self.scores()
        else:
            scores = self.unnorm_scores()

        plot_scores_hist(scores.iloc[:, comp], comp=comp, **kwargs)

    def plot_scree(self, log=False, diff=False, nticks=10):
        """
        Makes a scree plot of the singular values.

        Parameters
        ----------
        log: bool
            Take log base 10 of singular values.

        diff: bool
            Plot difference of successive singular values.

        nticks: int
            Number of tick marks to show.
        """
        scree_plot(self.svals_, log=log, diff=diff, nticks=10)

    def plot_var_expl_prop(self, nticks=10):
        """
        Plots the proportion of variance explained for each component.

        Parameters
        ----------
        nticks: int
            Number of tick marks to show.
        """
        plot_var_expl_prop(self.var_expl_prop_)

    def plot_var_expl_cum(self, nticks=10):
        """
        Plots the cumulative variance explained.

        Parameters
        ----------
        nticks: int
            Number of tick marks to show.
        """
        plot_var_expl_cum(self.var_expl_cum_)

    def plot_scores(self, norm=True,
                    start=0, n_components=3,  cats=None, cat_name=None,
                    dist_kws={}, scatter_kws={}):

        """
        Scores plot. See documentation of jive.viz.pca_scores_plot.scores_plot.

        Parameters
        ----------
        norm: bool
            Plot normalized scores.
        """

        if norm:
            scores = self.scores()
        else:
            scores = self.unnorm_scores()

        scores_plot(scores,
                    start=start,
                    ncomps=n_components,
                    cats=cats,
                    cat_name=cat_name,
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
        if norm:
            scores = self.scores()
        else:
            scores = self.unnorm_scores()
        scores = scores.iloc[:, comp]

        corr = np.corrcoef(scores, y)[0, 1]
        plt.scatter(scores, y)
        plt.xlabel('comp {} scores'.format(comp))
        plt.ylabel(ylabel)
        plt.title('correlation: {:1.4f}'.format(corr))

    def scores_corr_vs(self, y):
        """
        Computes the correlation between each PCA component and a continuous
        variable.
        """
        return np.array([np.corrcoef(self.scores_[:, i], y)[0, 1]
                         for i in range(self.n_components)])

    def plot_interactive_scores_slice(self, comp1, comp2, norm=True, cats=None):
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
        """
        if norm:
            scores = self.scores()
        else:
            scores = self.unnorm_scores()

        return interactive_slice(x=scores.iloc[:, comp1],
                                 y=scores.iloc[:, comp2],
                                 cats=cats,
                                 obs_names=self.obs_names_,
                                 xlab='component {}'.format(comp1),
                                 ylab='component {}'.format(comp2))

    # def interactive_scores_slice(self, comp1, comp2, cats=None):
    #     """
    #     model, saved_selected = pca.interactive_scores_slice(0, 1)
    #     model.to_df()
    #     """
    #     output_notebook()

    #     scores = self.scores

    #     source, model = setup_data(x=scores.iloc[:, comp1],
    #                                y=scores.iloc[:, comp2],
    #                                names=self.obs_names_,
    #                                classes=cats)

    #     figkwds = dict(plot_width=800, plot_height=800,
    #                    x_axis_label='component {}'.format(comp1),
    #                    y_axis_label='component {}'.format(comp2),
    #                    tools="pan,lasso_select,box_select,reset,help")

    #     handle = get_handle(source, figkwds)
    #     saved_selected = get_save_selected(handle, model)
    #     return model, saved_selected


# def safe_invert(x):
#     if np.allclose(x, 0):
#         return 0
#     else:
#         return 1.0/x

def _arg_checker(X, n_components):

    assert n_components is None or (n_components >= 1 and n_components <= min(X.shape))

    # extract data from X
    shape = X.shape

    if type(X) == pd.DataFrame:
        obs_names = np.array(X.index)
        var_names = np.array(X.columns)
    else:
        obs_names = range(X.shape[0])
        var_names = range(X.shape[1])

    return shape, obs_names, var_names


def pca_reconstruct(U, D, V, m):
    """
    Let the rank K pca of X be given by X ~= U D V^T. X in R^n x d
    where n = number of observations and d = number of variables.

    For a given set of scores returns the predicted reconstruction of X.
    For example, if u_i is the ith row of U (the scores for the
    ith observation) then this returns V D u_i + m.

    Parameters
    ---------
    u: the vector or matrix of scores (a vector in R^K or N x K matrix)

    D: the singular values (a list of length K)

    V: the loadings (nd.array of dimension d x K)

    m: the mean of the data (vector in R^d)
    """

    U = np.array(U)
    if U.ndim == 1:  # if U is a vector, then return as a vector
        is_vec = True
    else:
        is_vec = False

    if is_vec or U.shape[1] == 1:
        UD = U.reshape(1, -1) * D
    else:
        UD = U * D

    R = np.dot(UD, V) + m

    if is_vec:
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
