import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.robust.scale import mad
import pandas as pd

from bokeh.io import output_notebook  # , push_notebook, show
from jive.viz.scatter_plot_save_selected import setup_data, \
    get_handle, get_save_selected


def jitter_hist(x, **kwargs):
    """
    Jitter plot histogram
    """
    n, bins, patches = plt.hist(x, zorder=0, **kwargs)
    y = np.random.uniform(low=.05 * max(n), high=.1 * max(n), size=len(x))
    plt.scatter(x, y, color='red', zorder=1, s=1)


def qqplot(x, loc='mean', scale='std'):
    if loc == 'mean':
        mu_hat = np.mean(x)
    elif loc == 'median':
        mu_hat = np.median(x)

    if scale == 'std':
        sigma_hat = np.std(x)
    elif scale == 'mad':
        sigma_hat = mad(x)

    sm.qqplot(np.array(x), loc=mu_hat, scale=sigma_hat, line='s')


def plot_loading(v, abs_sorted=True, show_var_names=True,
                 significant_vars=None, show_top=None):
    """
    Plots a single loadings component.

    Parameters
    ----------
    v: array-like
        The loadings component.

    abs_sorted: bool
        Whether or not to sort components by their absolute values.


    significant_vars: {array-like, None}
        Indicated which features are significant in this component.

    show_top: {None, array-like}
        Will only display this number of top loadings components when
        sorting by absolute value.
    """
    if type(v) != pd.Series:
        v = pd.Series(v, index=['feature {}'.format(i) for i in range(len(v))])
        if significant_vars is not None:
            significant_vars = v.index.iloc[significant_vars]

    if abs_sorted:
        v_abs_sorted = np.abs(v).sort_values()
        v = v[v_abs_sorted.index]

        if show_top is not None:
            v = v[-show_top:]

            if significant_vars is not None:
                significant_vars = significant_vars[-show_top:]

    inds = np.arange(len(v))

    signs = v.copy()
    signs[v > 0] = 'pos'
    signs[v < 0] = 'neg'
    if significant_vars is not None:
        signs[v.index.difference(significant_vars)] = 'zero'
    else:
        signs[v == 0] = 'zero'
    s2c = {'pos': 'blue', 'neg': 'red', 'zero': 'grey'}
    colors = signs.apply(lambda x: s2c[x])

    # plt.figure(figsize=[5, 10])
    plt.scatter(v, inds, color=colors)
    plt.axvline(x=0, alpha=.5, color='black')
    plt.xlabel('loading value')
    if show_var_names:
        plt.yticks(inds, v.index)

    max_abs = np.abs(v).max()
    plt.xlim(-1.2 * max_abs, 1.2 * max_abs)

    for t, c in zip(plt.gca().get_yticklabels(), colors):
        t.set_color(c)
        if c != 'grey':
            t.set_fontweight('bold')

    # if comp is not None: # TODO: kill this
    #     plt.title('loading component {}'.format(comp))


def plot_scores_hist(s, comp, **kwargs):
    jitter_hist(s, **kwargs)
    if comp is not None:
        plt.title('component {} scores'.format(comp))


# def scree_plot(sv, log=False, diff=False, title='', nticks=10):
#     """
#     Makes a scree plot
#     """
#     ylab = 'singular value'

#     # possibly log values
#     if log:
#         sv = np.log(sv)
#         ylab = 'log ' + ylab

#     # possibly take differences
#     if diff:
#         sv = np.diff(sv)
#         ylab = ylab + ' difference'

#     n = len(sv)

#     plt.scatter(range(n), sv)
#     plt.plot(range(n), sv)
#     plt.ylim([1.1*min(0, min(sv)), 1.1 * max(sv)])
#     plt.xlim([-.01 * n, (n - 1) * 1.1])
#     nticks = min(nticks, len(sv))
#     plt.xticks(int(n/nticks) * np.arange(nticks))
#     plt.xlabel('index')
#     plt.ylabel(ylab)
#     plt.title(title)


def interactive_slice(x, y, cats=None, obs_names=None, xlab='x', ylab='y'):
        """
        model, saved_selected = pca.interactive_scores_slice(0, 1)
        model.to_df()
        """
        output_notebook()
        assert len(x) == len(y)
        if obs_names is None: obs_names = list(range(len(x)))

        source, model = setup_data(x=x,
                                   y=y,
                                   names=obs_names,
                                   classes=cats)

        figkwds = dict(plot_width=800, plot_height=800,
                       x_axis_label=xlab,
                       y_axis_label=ylab,
                       tools="pan,lasso_select,box_select,reset,help")

        handle = get_handle(source, figkwds)
        saved_selected = get_save_selected(handle, model)
        return model, saved_selected
