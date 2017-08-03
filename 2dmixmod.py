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
    M = df['Mact'].values
    labels = df['stage'].values
    Aks = 0.114*Av #Cardelli+1989

    M_ks = m_ks - mu - Aks

    #Range: y:{7,10}, x:{-3, 0}
    '''There must be a better way...'''
    sel = np.where(M_ks < -0.5)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]
    labels = labels[sel]

    sel = np.where(M_ks > -2.5)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]
    labels = labels[sel]


    sel = np.where(m_ks < 16.)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]
    labels = labels[sel]


    sel = np.where(m_ks > 6.)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]
    labels = labels[sel]

    return M_ks[0::], m_ks[0::], labels[0::], M[0::]

if __name__ == '__main__':
    vals = [-1.63,-1.59,-1.627,-1.626]
    errs = [0.002,0.005,0.20,0.057]
    noms = ['MixMod 2d', 'Mixmod 3d', 'RANSAC', 'Hawkins+17']
    x = np.arange(2,10,2)

    # fig, ax = plt.subplots()
    # ax.errorbar(x,vals,yerr=errs, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    # ax.scatter(x,vals,s=5,c='w',zorder=1000)
    # ax.set_xticks(x)
    # ax.set_xticklabels(noms,fontsize=10)
    # ax.set_ylabel('Absolute Red Clump magnitude (TRILEGAL)')
    # fig.tight_layout()
    # plt.show()


    x, y, labels, M = get_values()
    xerr = np.abs(0.01 + np.random.normal(0, 1, len(x)) * 0.005)
    yerr = np.abs(0.1 + np.random.normal(0, 1, len(y)) * 0.05)

    fig, ax = plt.subplots()
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']

    for i in range(int(np.nanmax(labels))+1):
        ax.scatter(x[labels==i], y[labels==i], c=c[i], s=4,zorder=1000)
        ax.errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.show()

    '''Priors for RC data run'''
    #m, b, Q, M, V
    bounds = [(-1.8,-1.4), (0, 1), (10.,13.), (0.1, 4.)]
    p0 = np.array([-1.6, 0.5, 11.0, 2.0])

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
        return lp + ll, 0.

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

    labels_mc = ["$b$", "$Q$", "$M$", "$V$"]
    # truths = true_params + [true_frac, true_outliers[0], true_outliers[1]]
    corner.corner(sampler.flatchain, bins=35, labels=labels_mc)#, truths=truths);

    plt.show()

    '''Doing posteriors'''
    norm = 0.0
    post_prob = np.zeros(len(x))
    for i in range(sampler.chain.shape[1]):
        for j in range(sampler.chain.shape[0]):
            ll_fg = lnlike_fg(sampler.chain[j,i])
            ll_bg = lnlike_bg(sampler.chain[j,i])
            post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
            norm += 1
    post_prob /= norm

    rc = np.median(sampler.flatchain[:,0])
    err = np.std(sampler.flatchain[:,0])
    rcy = np.linspace(6,15,10)

    rcx = np.linspace(rc-err,rc+err,10)
    rcy1 = np.ones(rcx.shape) * 15
    rcy2 = np.ones(rcx.shape) * 7

    # Plot the data points.
    plt.errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    # Plot the (true) outliers.
    plt.scatter(x, y, marker="s", s=22, c=post_prob, cmap="Blues_r", vmin=0, vmax=1, zorder=1000)
    plt.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001)
    plt.title('Clump luminosity: '+str(rc))
    plt.ylim([6,15])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()


    fig, ax = plt.subplots()
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']

    for i in range(int(np.nanmax(labels))+1):
        ax.scatter(x[labels==i], y[labels==i], c=c[i], s=4,zorder=1000)
        ax.errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    ax.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001)
    # ax.fill_between(fraclong, lolong, uplong,interpolate=True, facecolor='cyan')
    # plt.plot(rcx,rcy,lw=2,c='r',zorder=1001)
    # plt.axvline(rc+err,linestyle='--')
    # plt.axvline(rc-err,linestyle='--')
    ax.set_title('Clump luminosity: '+str(rc))
    ax.set_ylim([6,15])
    plt.show()
