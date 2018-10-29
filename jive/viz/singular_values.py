import matplotlib.pyplot as plt
import numpy as np


def comp_plot(values, nticks=10):
    """
    Parameters
    ----------

    values (list):

    nticks (int): number of tick marks to show.
    """
    n = len(values)
    plt.scatter(range(n), values)
    plt.plot(range(n), values)

    plt.ylim([1.1*min(0, min(values)), 1.1 * max(values)])
    plt.xlim([-.01 * n, (n - 1) * 1.1])
    nticks = min(nticks, n)
    plt.xticks(int(n/nticks) * np.arange(nticks))


def plot_var_expl_prop(var_expl_prop, nticks=10):
    """
    Plots the proportion of variance explained for each PCA component

    Parameters
    ----------
    nticks (int): number of tick marks to show.
    """

    comp_plot(var_expl_prop, nticks=nticks)
    plt.ylabel('variance explained proportion')
    plt.ylim([0, 1])
    plt.xlabel('component index')


def plot_var_expl_cum(var_expl_cum, nticks=10):
    """
    Plots the cumulative variance explained for PCA components.

    Parameters
    ----------
    nticks (int): number of tick marks to show.
    """
    comp_plot(var_expl_cum)
    plt.ylabel('cumulative variance explained')
    plt.ylim([0, 1])
    plt.xlabel('component index')


def scree_plot(sv, log=False, diff=False, nticks=10):
    """
    Makes a scree plot of the singular values.

    Parameters
    ----------
    log (bool): take log base 10 of singular values.

    diff (bool): plot difference of successive singular values.

    nticks (int): number of tick marks to show.
    """
    ylab = 'singular value'

    # possibly log values
    if log:
        sv = np.log10(sv)
        ylab = 'log ' + ylab

    # possibly take differences
    if diff:
        sv = np.diff(sv)
        ylab = ylab + ' difference'

    comp_plot(sv, nticks=nticks)
    plt.xlabel('index')
    plt.ylabel(ylab)
