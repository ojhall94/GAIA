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

def get_values():
    sfile = glob.glob('../data/TRILEGAL_sim/*.all.*.txt')[0]
    # sfile = glob.glob('../data/TRILEGAL/*.dat')[0]
    df = pd.read_csv(sfile, sep='\s+')

    '''This function corrects for extinction and sets the RC search range'''
    m_ks = df['Ks'].values
    mu = df['m-M0'].values
    Av = df['Av'].values
    Aks = 0.114*Av #Cardelli+1989

    M_ks = m_ks - mu - Aks

    #Range: y:{7,10}, x:{-3, 0}
    '''There must be a better way...'''
    sel = np.where(M_ks < -0.25)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]

    sel = np.where(M_ks > -2.)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]

    sel = np.where(m_ks < 16.)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]

    sel = np.where(m_ks > 7.)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]

    return M_ks[0:2000], m_ks[0:2000]


if __name__ == '__main__':
    fake = False

    if fake:
        '''Getting the data'''
        true_frac = 0.8
        #Unit slope, zero intercept
        true_params = [1.0, 0.0]
        #Zero mean, unit variance
        true_outliers = [0.0,1.0]

        np.random.seed(12)
        x = np.sort(np.random.uniform(-2,2,1000))
        yerr = 0.2 * np.ones_like(x)
        y = true_params[0] * x + true_params[1] + yerr * np.random.randn(len(x))

        #Lets replace soem points with outliers
        m_bkg = np.random.rand(len(x)) > true_frac
        y[m_bkg] = true_outliers[0]
        y[m_bkg] += np.sqrt(true_outliers[1]**2) * np.random.randn(sum(m_bkg))

        x0 = np.linspace(-2.1, 2.1, 200)
        y0 = np.dot(np.vander(x0, 2), true_params)

        plt.errorbar(x, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
        plt.scatter(x[m_bkg], y[m_bkg], marker="s", s=22, c="w", zorder=1000)
        plt.scatter(x[~m_bkg], y[~m_bkg], marker="o", s=22, c="k", zorder=1000)
        plt.plot(x0, y0, color="k", lw=1.5)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.ylim(-2.5, 2.5)
        plt.xlim(-2.1, 2.1)
        plt.show()

        '''Priors for DFM test run'''
        bounds = [(0.1, 1.9), (-0.9, 0.9), (0, 1), (-2.4, 2.4), (0., 2.)]
        p0 = np.array([1.0, 0.0, 0.7, 0.0, 1.0])


    if fake == False:
        x, y = get_values()
        xerr = 0.05 + np.random.normal(0, 1, len(x)) * 0.005
        yerr = np.abs(0.1 + np.random.normal(0, 1, len(y)) * 0.05)
        plt.scatter(x, y, c="b", s=4,zorder=1000)
        # plt.errorbar(x, y, yerr=yerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
        # plt.plot(x0, y0, color="k", lw=1.5)
        plt.xlabel("$x$")
        plt.ylabel("$y$")
        plt.show()

        '''Priors for RC data run'''
        #m, b, Q, M, V
        bounds = [(-1.8,-1.4), (0, 1), (10.,13.), (0.1, 4.)]
        p0 = np.array([-1.6, 0.5, 11.0, 2.0])

    '''Running the MCMC'''

    # Define the probabilistic model...
    # A simple prior:


    def lnprior(p):
        # We'll just put reasonable uniform priors on all the parameters.
        if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
            return -np.inf
        return 0

    # The "foreground" linear likelihood:
    def lnlike_fg(p):
        b, _, M, V = p
        model = np.zeros(y.shape) + b
        return -0.5 * ((model - x) / xerr)**2 - np.log(xerr)

    # The "background" outlier likelihood:
    def lnlike_bg(p):
        _, Q, M, V = p
        return -0.5 * ((M - y)/ V)**2 - np.log(V)

    # Full probabilistic model.
    def lnprob(p):
        b, Q, M, V = p

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
    ndim, nwalkers = 4, 32
    p0 = [p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]

    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Run a burn-in chain and save the final location.
    pos, _, _, _ = sampler.run_mcmc(p0, 500)

    # Run the production chain.
    sampler.reset()
    sampler.run_mcmc(pos, 1500);

    labels = ["$b$", "$Q$", "$M$", "$V$"]
    # truths = true_params + [true_frac, true_outliers[0], true_outliers[1]]
    corner.corner(sampler.flatchain, bins=35, labels=labels)#, truths=truths);

    plt.show()

    '''Doing posteriors'''
    norm = 0.0
    post_prob = np.zeros(len(x))
    for i in range(sampler.chain.shape[1]):
        for j in range(sampler.chain.shape[0]):
            ll_fg, ll_bg = sampler.blobs[i][j]
            post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
            norm += 1
    post_prob /= norm

    # Plot the data points.
    plt.errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    # Plot the (true) outliers.
    plt.scatter(x, y, marker="s", s=22, c=post_prob, cmap="Blues_r", vmin=0, vmax=1, zorder=1000)

    # Plot the (true) good points.
    # plt.scatter(x[~m_bkg], y[~m_bkg], marker="o", s=22, c=post_prob[~m_bkg], cmap="Blues_r", vmin=0, vmax=1, zorder=1000)

    # Plot the true line.
    # plt.plot(x0, y0, color="k", lw=1.5)

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()

    plt.scatter(x, y, c=post_prob, cmap="Blues_r", s=4,zorder=1000)
    plt.show()






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
