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

    corrections = pd.DataFrame(columns=['M_ks<','M_ks>','Ks<','Ks>','Mact<','M/H>','M/H<'])
    corr = [-0.5, -2.5, 15., 6., 1.4, -.5, .5]
    corrections.loc[0] = corr
    corrections.to_csv('../Output/data_selection.csv')

    #Set selection criteria
    df = df[df.M_ks < corr[0]]
    df = df[df.M_ks > corr[1]]

    df = df[df.Ks < corr[2]]
    df = df[df.Ks > corr[3]]

    df = df[df.Mact < corr[4]]

    df = df[df['[M/H]'] > corr[5]]
    df = df[df['[M/H]'] < corr[6]]

    # df = df[df.stage == 4]
    df = df[0:10000]

    df = get_errors(df)

    return df.M_ks.values, df.Ks.values, df.stage, df

def get_errors(df):
    DR = 2  #Choosing the data release
    if DR == 1:
        df['pi_err'] = np.abs(0.3 + np.random.normal(0,0.5,len(df)) * 0.3) #mas

    if DR == 2:
        df['pi_err'] = np.abs(10.e-3 + np.random.normal(0,0.5,len(df)) * 2.e-3) #mas

    df['d'] = 10**(1+ (df['m-M0']/5))       #Getting all distances (pc)
    df['pi'] = 1000/df['d']                 #Getting all parallax (mas)
    df['sig_d'] = (1000*df['pi_err']/df['pi']**2)  #Propagating the Gaia error

    df['sig_mu'] = np.sqrt( (5*np.log10(np.e)/df['d'])**2 * df['sig_d']**2 )
    df['sig_M'] = df['sig_mu']  #Until we incorporate errors on Aks and Ks

    plt.errorbar(df['pi'],df['Ks'],xerr=df['pi_err'],alpha=.1,fmt=".k",c='grey',zorder=999)
    plt.scatter(df['pi'],df['Ks'],s=5,zorder=1000)
    plt.show()

    plt.errorbar(df['d'],df['Ks'],xerr=df['sig_d'],alpha=.1,fmt=".k",c='grey',zorder=999)
    plt.scatter(df['d'],df['Ks'],s=5,zorder=1000)
    plt.show()
    return df

def lnprior(p):
    # We'll just put reasonable uniform priors on all the parameters.
    if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
        return -np.inf
    return 0

# The "foreground" linear likelihood:
def lnlike_fg(p):
    b, sigrc, _, o, sigo = p
    sig = np.sqrt(sigrc**2 + xerr**2)

    return -0.5 * ((x - b) / sig)**2 - np.log(sig)

# The "background" outlier likelihood:
def lnlike_bg(p):
    _, _, Q, o, sigo = p
    sig = np.sqrt(sigo**2 + xerr**2)
    val = np.abs(x).max() + np.abs(x).min()

    xn = x + val
    on = np.abs(o)

    return -np.log(xn) -np.log(sig) - 0.5 * (np.log(xn) - on)**2/sig**2

# Full probabilistic model.
def lnprob(p):
    b, sigrc, Q, o, sigo = p

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

def save_library(df, chain,labels_mc):
    results = pd.DataFrame(columns=labels_mc)
    stddevs = pd.DataFrame(columns=[l+'err' for l in labels_mc])

    npa = chain.shape[1]
    r = np.zeros(npa)
    s = np.zeros(npa)
    for idx in np.arange(npa):
        r[idx] = np.median(chain[:,idx])
        s[idx] = np.std(chain[:,idx])
    results.loc[0], stddevs.loc[0] = r, s

    output = pd.concat([results,stddevs],axis=1)
    output.to_csv('../Output/2dmix_results.csv')
    df.to_csv('../Output/2dmix_selected_data.csv')

    return results, stddevs

if __name__ == '__main__':


####---SETTING UP DATA
    x, y, labels, df = get_values()
    xerr = df['sig_M']
    # xerr = np.abs(0.05 + np.random.normal(0, 1, len(x)) * 0.005)

    # yerr = np.abs(0.1 + np.random.normal(0, 1, len(y)) * 0.05)

####---PLOTTING INITIAL DATA
    fig, ax = plt.subplots()
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    for i in range(int(np.nanmax(labels))+1):
        ax.scatter(x[labels==i], y[labels==i], c=c[i], s=4,zorder=1000)
        ax.errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    plt.savefig('../Output/dataset.png')
    plt.show()

####---SETTING UP AND RUNNING MCMC
    labels_mc = ["$b$", r"$\sigma(RC)$", "$Q$", "$o$", r"$\sigma(o)$"]
    bounds = [(-1.7,-1.4), (0.01,0.2), (0, 1), (0.0,2.0), (0.1, 2.)]
    start_params = np.array([-1.6, 0.1, 0.5, 0.1, 1.0])

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
    for p1, lnpp, lnlp in tqdm(sampler.sample(p0, iterations=1000)):
        pass

    # Run the production chain.
    print('Running emcee...')
    sampler.reset()
    for pp, lnpp, lnlp in tqdm(sampler.sample(p1, iterations=500)):
        pass
    chain = sampler.chain[0,:,:,:].reshape((-1, ndims))

####---CONSOLIDATING RESULTS
    corner.corner(chain, bins=35, labels=labels_mc)
    plt.savefig('../Output/corner.png')
    plt.show()

    print('Calculating posteriors...')
    norm = 0.0
    fg_pp = np.zeros(len(x))
    bg_pp = np.zeros(len(x))
    lotemp = sampler.chain[0,:,:,:]

    for i in tqdm(range(lotemp.shape[0])):
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

####---PLOTTING RESULTS
    print('Plotting results...')
    rc = np.median(chain[:,0])  #RC luminosity
    err = np.std(chain[:,0])    #stddev on RC luminosity
    rcy = np.linspace(6,15,10)  #Y-axis for RC plot

    rcx = np.linspace(rc-err,rc+err,10) #Setting up shading bounds
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
    plt.savefig('../Output/posteriors.png')
    plt.show()

    #Plot labeled results
    fig, ax = plt.subplots()
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    for i in range(int(np.nanmax(labels))+1):
        ax.scatter(x[~mask][labels[~mask]==i], y[~mask][labels[~mask]==i], c=c[i], s=4,zorder=1000)
        ax.errorbar(x[~mask], y[~mask], xerr=xerr[~mask], fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    ax.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001)
    ax.set_title('Clump luminosity: '+str(rc))
    ax.set_ylim([6,15])
    plt.savefig('../Output/labeled_results.png')
    plt.show()



####---SAVING STUFF INTO A PANDAS LIBRARY
    print('Saving output...')
    results, stddevs = save_library(df,chain,labels_mc)

    plt.scatter(x,np.exp(lnlike_fg(results.loc[0].values)))
    plt.savefig('../Output/fg_like.png')
    plt.close()
    plt.scatter(x, np.exp(lnlike_bg(results.loc[0].values)))
    plt.savefig('../Output/bg_like.png')
    plt.close()
#__________________GETTING CORRECTION____________________________
    print('Calculating corrections...')
    #Plot showing correction
    m_ks = df.Ks.values[mask]
    Aks = df.Aks.values[mask]
    pi_o = df['pi'].values[mask]
    pi_o_err = df['pi_err'].values[mask]

    #Calculating RC parallax from our fit value for magnitude
    Mrc = np.ones(y[mask].shape)*rc                 #Fit RC luminosity
    Mrcerr = np.ones(y[mask].shape)*err             #Fit RC error
    mu_rc = m_ks - Mrc - Aks                        #Calculating distance modulus

    d_rc = 10**(1+mu_rc/5)      #Calculating distance
    pi_rc = 1000/d_rc            #Calculating parallax

    d_err = np.sqrt( (2*np.log(10)*10**(mu_rc/5))**2 * Mrcerr**2 )
    pi_rc_err = 1000 * d_err / d_rc**2

    #Fit a straight line to the data
    fit = np.polyfit(pi_rc, pi_o, 1)
    fn = np.poly1d(fit)

    #Plotting results
    fig, ax = plt.subplots(2)
    ax[0].errorbar(pi_rc,pi_o, xerr=pi_rc_err, yerr=pi_o_err,fmt=",k",alpha=.1, ms=0, capsize=0, lw=1, zorder=999)
    im = ax[0].scatter(pi_rc,pi_o, marker="s", s=10, c=m_ks, cmap="viridis", zorder=1000)
    ax[0].plot(pi_rc,pi_rc,linestyle='--',c='r',alpha=.5, zorder=1001, label='One to One')
    ax[0].plot(pi_rc, fn(pi_rc),c='k',alpha=.9, zorder=1002, label='Straight line polyfit')
    ax[0].set_xlabel('RC parallax')
    ax[0].set_ylabel('TRILEGAL parallax')
    ax[0].set_title(r'RC mixture model fit compared to TRILEGAL parallax (M $<$ 1.4Msol, -.5 $<$ [M/H] $<$ .5)')
    ax[0].legend(loc='best')

    ax[1].errorbar(pi_rc,pi_o-pi_rc, xerr=pi_rc_err,yerr=np.sqrt(pi_o_err**2 + pi_rc_err**2), fmt=",k",alpha=.2, ms=0, capsize=0, lw=1, zorder=999)
    ax[1].scatter(pi_rc,pi_o - pi_rc, s=10,c=m_ks,cmap='viridis',zorder=1000)
    ax[1].set_xlabel('RC parallax')
    ax[1].set_ylabel('TRILEGAL parallax - RC parallax')
    ax[1].set_title(r'Residuals to one-to-one for upper plot')
    ax[1].axhline(y=0.0,c='r',alpha=.5,linestyle='--')
    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
    norm = matplotlib.colors.Normalize(vmin=m_ks.min(),vmax=m_ks.max())
    col = matplotlib.colorbar.ColorbarBase(cbar_ax,cmap='viridis',norm=norm,orientation='vertical',label='TRILEGAL apaprent magnitude')

    plt.savefig('../Output/corrections.png')
    plt.show()
