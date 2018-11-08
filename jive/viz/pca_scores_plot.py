import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def scores_plot(values, start=0, ncomps=3,
                classes=None, class_name=None,
                dist_kws={}, scatter_kws={}):
    """


    """
    ncomps = min(ncomps, values.shape[1])

    if type(values) == pd.DataFrame:
        values_ = values.iloc[:, start:(start + ncomps)].copy()
    else:
        values_ = pd.DataFrame(np.array(values)[:, start:(start + ncomps)])

    if classes is not None:
        assert len(classes) == values_.shape[0]
        if hasattr(classes, 'name'):
            class_name = classes.name
        elif class_name is None:
            class_name = 'classes'

        values_[class_name] = np.array(classes)
        values_[class_name] = values_[class_name].astype(str)
    else:
        class_name = None

    g = sns.PairGrid(values_, hue=class_name,
                     vars=list(values_.columns.difference([class_name])))  # Hack
    g = g.map_upper(plt.scatter, **scatter_kws)
    g = g.map_diag(sns.distplot, rug=True, **dist_kws)

    # kill lower diagonal plots
    for i, j in zip(*np.triu_indices_from(g.axes, 1)):
        g.axes[j, i].set_visible(False)

    if classes is not None:
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
