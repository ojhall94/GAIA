# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Oliver J. Hall

import numpy as np
import pandas as pd

import glob
import sys
import corner as corner
from tqdm import tqdm
import seaborn as sns
import emcee

import matplotlib
# import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

from statsmodels.robust.scale import mad as mad_calc
from sklearn import linear_model
import scipy.stats as stats
import scipy.misc as misc

if __name__ == '__main__':


    true_frac = 0.8
    #Unit slope, zero intercept
    true_params = [1.0, 0.0]
    #Zero mean, unit variance
    true_outliers = [0.0,1.0]

    np.random.seed(12)
    x = np.sort(np.random.uniform(-2,2,100))
    yerr = 0.2 * np.ones_like(x)
    y = true_params[0] * x + true_params[1] + yerr * np.random.randn(len(x))

    #Lets replace soem points with outliers
    m_bkg = np.random.rand(len(x)) > true_frac
    y[m_bkg] = true_outliers[0]
    y[m_bkg] += np.sqrt(true_outliers[1]+yerr[m_bkg]**2) * np.random.randn(sum(m_bkg))

    plt.errorbar(x, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    plt.scatter(x[m_bkg], y[m_bkg], marker="s", s=22, c="w", zorder=1000)
    plt.scatter(x[~m_bkg], y[~m_bkg], marker="o", s=22, c="k", zorder=1000)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.ylim(-2.5, 2.5)
    plt.xlim(-2.1, 2.1)
    plt.show()

    '''not written up the below'''

    # Define the probabilistic model...
    # A simple prior:
    bounds = [(0.1, 1.9), (-0.9, 0.9), (0, 1), (-2.4, 2.4), (-7.2, 5.2)]
    def lnprior(p):
        # We'll just put reasonable uniform priors on all the parameters.
        if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
            return -np.inf
        return 0

    # The "foreground" linear likelihood:
    def lnlike_fg(p):
        m, b, _, M, lnV = p
        model = m * x + b
        return -0.5 * (((model - y) / yerr) ** 2 + 2 * np.log(yerr))

    # The "background" outlier likelihood:
    def lnlike_bg(p):
        _, _, Q, M, lnV = p
        var = np.exp(lnV) + yerr**2
        return -0.5 * ((M - y) ** 2 / var + np.log(var))

    # Full probabilistic model.
    def lnprob(p):
        m, b, Q, M, lnV = p

        # First check the prior.
        lp = lnprior(p)
        if not np.isfinite(lp):
            return -np.inf, None

        # Compute the vector of foreground likelihoods and include the q prior.
        ll_fg = lnlike_fg(p)
        arg1 = ll_fg + np.log(Q)

        # Compute the vector of background likelihoods and include the q prior.
        ll_bg = lnlike_bg(p)
        arg2 = ll_bg + np.log(1.0 - Q)

        # Combine these using log-add-exp for numerical stability.
        ll = np.sum(np.logaddexp(arg1, arg2))

        # We're using emcee's "blobs" feature in order to keep track of the
        # foreground and background likelihoods for reasons that will become
        # clear soon.
        return lp + ll, (arg1, arg2)

    # Initialize the walkers at a reasonable location.
    ndim, nwalkers = 5, 32
    p0 = np.array([1.0, 0.0, 0.7, 0.0, np.log(2.0)])
    p0 = [p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]

    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Run a burn-in chain and save the final location.
    pos, _, _, _ = sampler.run_mcmc(p0, 500)

    # Run the production chain.
    sampler.reset()
    sampler.run_mcmc(pos, 1500);

    labels = ["$m$", "$b$", "$Q$", "$M$", "$\ln V$"]
    truths = true_params + [true_frac, true_outliers[0], np.log(true_outliers[1])]
    corner.corner(sampler.flatchain, bins=35, extents=bounds, labels=labels, truths=truths);






    # sig = 0.138
    # M = -1.62
    #
    # x = np.linspace(-2.,-0.25,10000)
    #
    # F = (1/(np.sqrt(2*np.pi) * sig)) * np.exp(-(x - M)**2/(2*sig**2))
    # Fmax = (1/(np.sqrt(2*np.pi) * sig))
    # F/=Fmax
    #
    # plt.plot(x,F,linewidth=2)
    # plt.plot(x,F/2)
    # plt.plot(x,F/4)
    # plt.plot(x,F/8)
    # plt.plot(x,F/16)
    # plt.show()
    #
    #
    # xp = np.linspace(10,7,1000)
    # l = 1.
    # P = np.exp(-l) * l**xp / misc.factorial(xp)
    # m = P.max()
    #
    # for l in range(0,10):
    #     P = np.exp(-l) * l**xp / misc.factorial(xp)
    #     P /= m
    #     plt.plot(xp,P)
    # plt.show()
