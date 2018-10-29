import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def scores_plot(values, start=0, ncomps=3,
                cats=None, cat_name=None,
                dist_kws={}, scatter_kws={}):
    """


    """
    ncomps = min(ncomps, values.shape[1])

    if type(values) == pd.DataFrame:
        values_ = values.iloc[:, start:(start + ncomps)].copy()
    else:
        values_ = pd.DataFrame(np.array(values)[:, start:(start + ncomps)])

    if cats is not None:
        assert len(cats) == values_.shape[0]
        if cat_name is None:
            cat_name = 'cat'
        values_[cat_name] = np.array(cats)
    else:
        cat_name = None

    g = sns.PairGrid(values_, hue=cat_name)
    g = g.map_upper(plt.scatter, **scatter_kws)
    g = g.map_diag(sns.distplot, rug=True, **dist_kws)

    # kill lower diagonal plots
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[j, i].set_visible(False)

    if cats is not None:
        g.add_legend()

    # label axes properly
    xlabels, ylabels = [], []
    for ax in g.axes[-1, :]:
        xlabel = ax.xaxis.get_label_text()
        xlabels.append(xlabel)
    for ax in g.axes[:, 0]:
        ylabel = ax.yaxis.get_label_text()
        ylabels.append(ylabel)
    for i in range(len(xlabels)):
        for j in range(len(ylabels)):
            if i == j:
                g.axes[j, i].xaxis.set_label_text(xlabels[i])
                g.axes[j, i].yaxis.set_label_text(ylabels[j])
