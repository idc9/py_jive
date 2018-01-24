import numpy as np
import matplotlib.pyplot as plt

from statsmodels.distributions.empirical_distribution import ECDF

def plot_joint_diagnostic(joint_svsq,
                           wedin_sv_samples,
                           random_sv_samples,
                           wedin_percentile=95,
                           random_percentile=5):

    plt.figure(figsize=[10, 10])

    # compute sv_threshold
    wedin_cutoff = np.percentile(wedin_sv_samples, wedin_percentile)
    rand_cutoff = np.percentile(random_sv_samples, random_percentile)
    # svsq_cutoff = max(rand_cutoff, wedin_cutoff)
    if rand_cutoff > wedin_cutoff:
        svsq_cutoff = rand_cutoff
        rand_lw = 3
        wedin_lw = 1
    else:
        svsq_cutoff = wedin_cutoff
        rand_lw = 1
        wedin_lw = 3

    joint_rank_est = sum(joint_svsq > svsq_cutoff)

    # plot wedin CDF
    wedin_xvals, wedin_cdf = get_cdf_vals(wedin_sv_samples)

    plt.plot(wedin_xvals,
             wedin_cdf,
             color='blue',
             ls='dotted',
             label='wedin cdf')

    plt.axvline(wedin_cutoff,
                color='blue',
                ls='dashed',
                lw=wedin_lw,
                label = 'wedin %dth percentile' % wedin_percentile)

    # plot random CDF
    rand_xvals, rand_cdf = get_cdf_vals(random_sv_samples)
    plt.plot(rand_xvals,
             1 - rand_cdf,
             color='red',
             ls='dotted',
             label='random survival')

    plt.axvline(rand_cutoff,
                color='red',
                ls='dashed',
                lw=rand_lw,
                label = 'random %dth percentile' % random_percentile)

    # plot joint singular values
    first_joint = True
    first_nonjoint = True
    for sv_sq in joint_svsq:

        if sv_sq > svsq_cutoff:

            label = 'joint singular value' if first_joint else ''
            first_joint = False

            plt.axvline(sv_sq,
                        color='black',
                        label=label,
                        lw=3)
        else:

            label = 'nonjoint singular value' if first_nonjoint else ''
            first_nonjoint = False
            plt.axvline(sv_sq,
                        color='grey',
                        label=label,
                        lw=1)

    plt.xlabel('squared singluar value')
    plt.legend()
    plt.ylim([0, 1])
    plt.xlim(xmin=0)
    plt.title('joint singluar value thresholding (joint rank estimate = %d)' % joint_rank_est)

def get_cdf_vals(samples):
    ecdf = ECDF(samples)
    xvals = np.linspace(start=min(samples), stop=max(samples), num=100)
    cdf_vals = np.array([ecdf(x) for x in xvals])


    return xvals, cdf_vals
