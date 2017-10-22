import numpy as np
import pandas as pd

import seaborn.apionly as sns
import matplotlib.pyplot as plt


def generate_data_ajive_fig2(seed=None):
    """
    Samples the data from AJIVE figure 2. Note here we use rows as observations
    i.e. data matrices are n x d where n = # observations.

    X_obs, X_joint, X_indiv, X_noise, Y_obs, Y_joint, Y_indiv, Y_noise =
    generate_data_ajive_fig2()
    """
    # TODO: return ndarray instead of matrix
    if seed:
        np.random.seed(seed)

    # Sample X data
    X_joint = np.bmat([[np.ones((50, 50))],
                      [-1*np.ones((50, 50))]])
    X_joint = 5000 * np.bmat([X_joint, np.zeros((100, 50))])

    X_indiv = 5000 * np.bmat([[-1 * np.ones((25, 100))],
                              [np.ones((25, 100))],
                              [-1 * np.ones((25, 100))],
                              [np.ones((25, 100))]])

    X_noise = 5000 * np.random.normal(loc=0, scale=1, size=(100, 100))

    X_obs = X_joint + X_indiv + X_noise

    # Sample Y data
    Y_joint = np.bmat([[-1 * np.ones((50, 2000))],
                       [np.ones((50, 2000))]])
    Y_joint = np.bmat([np.zeros((100, 8000)), Y_joint])

    Y_indiv_t = np.bmat([[np.ones((20, 5000))],
                        [-1 * np.ones((20, 5000))],
                        [np.zeros((20, 5000))],
                        [np.ones((20, 5000))],
                        [-1 * np.ones((20, 5000))]])

    Y_indiv_b = np.bmat([[np.ones((25, 5000))],
                        [-1 * np.ones((50, 5000))],
                        [np.ones((25, 5000))]])

    Y_indiv = np.bmat([Y_indiv_t, Y_indiv_b])

    Y_noise = np.random.normal(loc=0, scale=1, size=(100, 10000))

    Y_obs = Y_joint + Y_indiv + Y_noise

    # TODO: make this into a list of dicts i.e. hierarchical
    return X_obs, X_joint, X_indiv, X_noise, Y_obs, Y_joint, Y_indiv, Y_noise


def np_matrix_to_pd_dataframe(mat):

    df = pd.DataFrame(mat)
    # df.index = range(1, mat.shape[0] + 1)
    # df.columns = range(1, mat.shape[1] + 1)
    df.index.name = 'observations'
    df.columns.name = 'features'

    return df


def plot_ajive_fig2(X_obs, X_joint, X_indiv, X_noise,
                    Y_obs, Y_joint, Y_indiv, Y_noise):
    """
    Plots figure 2 from AJIVE
    """
    plt.figure(figsize=[10, 20])

    # Observed
    plt.subplot(4, 2, 1)
    sns.heatmap(np_matrix_to_pd_dataframe(X_obs), xticklabels=20,
                yticklabels=20)
    plt.title('X observed')

    plt.subplot(4, 2, 2)
    sns.heatmap(np_matrix_to_pd_dataframe(Y_obs), xticklabels=2000,
                yticklabels=20)
    plt.title('Y observed')

    # joint
    plt.subplot(4, 2, 3)
    sns.heatmap(X_joint, xticklabels=20, yticklabels=20)
    plt.title('X Joint')

    plt.subplot(4, 2, 4)
    sns.heatmap(Y_joint,  xticklabels=2000, yticklabels=20)
    plt.title('Y Joint')

    # individual
    plt.subplot(4, 2, 5)
    sns.heatmap(X_indiv,xticklabels=20, yticklabels=20)
    plt.title('X individual')

    plt.subplot(4, 2, 6)
    sns.heatmap(Y_indiv,  xticklabels=2000, yticklabels=20)
    plt.title('Y individual')

    # Noise
    plt.subplot(4, 2, 7)
    sns.heatmap(X_noise, xticklabels=20, yticklabels=20)
    plt.title('X noise')

    plt.subplot(4, 2, 8)
    sns.heatmap(Y_noise,  xticklabels=2000, yticklabels=20)
    # cbar_kws={"orientation": "horizontal"}
    plt.title('Y noise')


def plot_jive_results_2blocks(X_obs, J_x, I_x, E_x,
                              Y_obs, J_y, I_y, E_y):
    """
    Heat map of JIVE results
    """

    plt.figure(figsize=[10, 20])

    # Observed
    plt.subplot(4, 2, 1)
    sns.heatmap(np_matrix_to_pd_dataframe(X_obs), xticklabels=20,
                yticklabels=20)
    plt.title('X observed')

    plt.subplot(4, 2, 2)
    sns.heatmap(np_matrix_to_pd_dataframe(Y_obs), xticklabels=2000,
                yticklabels=20)
    plt.title('Y observed')

    # joint
    plt.subplot(4, 2, 3)
    sns.heatmap(J_x, xticklabels=20, yticklabels=20)
    plt.title('X Joint (estimated)')

    plt.subplot(4, 2, 4)
    sns.heatmap(J_y,  xticklabels=2000, yticklabels=20)
    plt.title('Y Joint (estimated)')

    # individual
    plt.subplot(4, 2, 5)
    sns.heatmap(I_x, xticklabels=20, yticklabels=20)
    plt.title('X individual (estimated)')

    plt.subplot(4, 2, 6)
    sns.heatmap(I_y,  xticklabels=2000, yticklabels=20)
    plt.title('Y individual (estimated)')

    # Noise
    plt.subplot(4, 2, 7)
    sns.heatmap(E_x, xticklabels=20, yticklabels=20)
    plt.title('X noise (estimated)')

    plt.subplot(4, 2, 8)
    sns.heatmap(E_y,  xticklabels=2000, yticklabels=20)
    # cbar_kws={"orientation": "horizontal"}
    plt.title('Y noise (estimated)')


def threshold(x, epsilon=1e-10):
    x[abs(x) < epsilon] = 0
    return x


def plot_jive_residuals_2block(X_joint, X_indiv, X_noise, J_x, I_x, E_x,
                               Y_joint, Y_indiv, Y_noise, J_y, I_y, E_y):

    plt.figure(figsize=[10, 15])

    # compute residuals
    epsilon = 1e-8
    R_joint_x = threshold(X_joint - J_x, epsilon)
    R_indiv_x = threshold(X_indiv - I_x, epsilon)
    R_noise_x = threshold(X_noise - E_x, epsilon)

    R_joint_y = threshold(Y_joint - J_y, epsilon)
    R_indiv_y = threshold(Y_indiv - I_y, epsilon)
    R_noise_y = threshold(Y_noise - E_y, epsilon)

    # joint
    plt.subplot(3, 2, 1)
    sns.heatmap(R_joint_x, xticklabels=20, yticklabels=20)
    plt.title('X Joint (residuals)')

    plt.subplot(3, 2, 2)
    sns.heatmap(R_joint_y,  xticklabels=2000, yticklabels=20)
    plt.title('Y Joint (residuals)')

    # individual
    plt.subplot(3, 2, 3)
    sns.heatmap(R_indiv_x, xticklabels=20, yticklabels=20)
    plt.title('X individual (residuals)')

    plt.subplot(3, 2, 4)
    sns.heatmap(R_indiv_y,  xticklabels=2000, yticklabels=20)
    plt.title('Y individual (residuals)')

    # Noise
    plt.subplot(3, 2, 5)
    sns.heatmap(R_noise_x, xticklabels=20, yticklabels=20)
    plt.title('X noise (residuals)')

    plt.subplot(3, 2, 6)
    sns.heatmap(R_noise_y,  xticklabels=2000, yticklabels=20)
    plt.title('Y noise (residuals)')
