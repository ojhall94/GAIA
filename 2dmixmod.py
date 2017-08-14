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
    # sfile = glob.glob('../data/TRILEGAL_sim/*.all.*.txt')[0]
    sfile = glob.glob('../data/TRILEGAL_sim/*all*.txt')[0]
    df = pd.read_csv(sfile, sep='\s+')

    '''This function corrects for extinction and sets the RC search range'''
    df['Aks'] = 0.114*df.Av #Cardelli+1989>-
    df['M_ks'] = df.Ks - df['m-M0'] - df.Aks

    #Set selection criteria
    df = df[df.M_ks < -0.5]
    df = df[df.M_ks > -2.5]

    df = df[df.Ks < 16.]
    df = df[df.Ks > 6.]

    df = df[df.Mact < 1.4]

    df = df[df['[M/H]'] > -.5]
    df = df[df['[M/H]'] < .5]

    # df = df[df.stage == 4]

    return df.M_ks, df.Ks, df.stage, df

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
    return lp + ll

if __name__ == '__main__':
    posteriors = False


    x, y, labels, df = get_values()
    xerr = np.abs(0.05 + np.random.normal(0, 1, len(x)) * 0.005)
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
    bounds = [(-1.7,-1.4), (0, 1), (10.,13.), (0.1, 4.)]
    start_params = np.array([-1.6, 0.5, 11.0, 2.0])

    # Initialize the walkers at a reasonable location.
    ntemps, ndims, nwalkers = 2, len(bounds), 32

    p0 = np.zeros([ntemps,nwalkers,ndims])
    for i in range(ntemps):
        for j in range(nwalkers):
            p0[i,j,:] = start_params * (1.0 + np.random.randn(ndims) * 0.0001)


    # Set up the sampler.
    sampler = emcee.PTSampler(ntemps, nwalkers, ndims, lnprob, lnprior,threads=2)

    # Run a burn-in chain and save the final location.
    print('Burning in emcee...')
    for p1, lnpp, lnlp in tqdm(sampler.sample(p0, iterations=500)):
        pass
    # pos, _, _, _ = sampler.run_mcmc(p0, 500)

    # Run the production chain.
    print('Running emcee...')
    sampler.reset()
    for pp, lnpp, lnlp in tqdm(sampler.sample(p1, iterations=1500)):
        pass
    # sampler.run_mcmc(pos, 1500)

    chain = sampler.chain[0,:,:,:].reshape((-1, ndims))

    labels_mc = ["$b$", "$Q$", "$M$", "$V$"]
    corner.corner(chain, bins=35, labels=labels_mc)
    plt.show()

    print('Calculating posteriors...')
    norm = 0.0
    fg_pp = np.zeros(len(x))
    bg_pp = np.zeros(len(x))
    lotemp = sampler.chain[0,:,:,:]

    for i in range(lotemp.shape[0]):
        for j in range(lotemp.shape[1]):
            ll_fg = lnlike_fg(lotemp[i,j])
            ll_bg = lnlike_bg(lotemp[i,j])
            fg_pp += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
            bg_pp += np.exp(ll_bg - np.logaddexp(ll_fg, ll_bg))
            norm += 1
    fg_pp /= norm
    bg_pp /= norm

    lnK = np.log(fg_pp) - np.log(bg_pp)
    mask = lnK > 1

    #Put together the shading on the plot
    rc = np.median(chain[:,0])
    err = np.std(chain[:,0])
    rcy = np.linspace(6,15,10)

    rcx = np.linspace(rc-err,rc+err,10)
    rcy1 = np.ones(rcx.shape) * 15
    rcy2 = np.ones(rcx.shape) * 7

    # Plot mixture model results
    plt.errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    plt.scatter(x, y, marker="s", s=22, c=fg_pp, cmap="Blues_r", vmin=0, vmax=1, zorder=1000)
    plt.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001)
    plt.title('Clump luminosity: '+str(rc))
    plt.ylim([6,15])
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()

    #Plot clarifying results
    fig, ax = plt.subplots()
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    for i in range(int(np.nanmax(labels))+1):
        ax.scatter(x[labels==i], y[labels==i], c=c[i], s=4,zorder=1000)
        ax.errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    ax.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001)
    ax.set_title('Clump luminosity: '+str(rc))
    ax.set_ylim([6,15])
    plt.show()

    '''Should probably build in the .pd readout here'''
#__________________GETTING CORRECTION____________________________


    #Plot showing correction
    m_ks = df.Ks.values[mask]
    mu = df['m-M0'].values[mask]
    Aks = df.Aks.values[mask]

    #Calculating Trilegal parallax
    d_o = 10**(1+mu/5)
    p_o = 1000/d_o

    #Calculating RC parallax
    Mrc = np.ones(y[mask].shape)*rc
    Mrcerr = np.ones(y[mask].shape)*err
    mu_rc = m_ks - Mrc - Aks
    sig_mu = np.sqrt(yerr[mask]**2 + xerr[mask]**2)

    d_rc = 10**(1+mu_rc/5)
    p_rc = 1000/d_rc
    sig_p = np.sqrt( (10**(1+mu_rc/5)*np.log(10)/5)**2 * sig_mu**2)

    oto = np.copy(p_rc)

    fit = np.polyfit(p_rc, p_o, 1)
    fn = np.poly1d(fit)

    fig, ax = plt.subplots(2)
    # plt.errorbar(p_rc,p_o, xerr=sig_p, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    im = ax[0].scatter(p_rc,p_o, marker="s", s=22, c=m_ks, cmap="viridis", zorder=1000)
    ax[0].plot(p_rc,oto,linestyle='--',c='r',alpha=.5, zorder=1001, label='One to One')
    ax[0].plot(p_rc, fn(p_rc),c='k',alpha=.9, zorder=1002, label='Straight line polyfit')
    ax[0].set_xlabel('RC parallax')
    ax[0].set_ylabel('TRILEGAL parallax')
    ax[0].set_title(r'RC mixture model fit compared to TRILEGAL parallax (M $<$ 1.4Msol, -.5 $<$ [M/H] $<$ .5)')
    ax[0].legend(loc='best')

    ax[1].scatter(p_rc,p_o - p_rc, c=m_ks,cmap='viridis')
    ax[1].set_xlabel('RC parallax')
    ax[1].set_ylabel('TRILEGAL parallax - RC parallax')
    ax[1].set_title(r'Residuals to one-to-one for upper plot')
    ax[1].axhline(y=0.0,c='r',alpha=.5,linestyle='--')
    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
    norm = matplotlib.colors.Normalize(vmin=m_ks.min(),vmax=m_ks.max())
    col = matplotlib.colorbar.ColorbarBase(cbar_ax,cmap='viridis',norm=norm,orientation='vertical',label='TRILEGAL apaprent magnitude')
    plt.show()
