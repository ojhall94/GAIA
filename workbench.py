# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Oliver J. Hall

import numpy as np
import pandas as pd

import glob
import sys
import corner as corner
from tqdm import tqdm
# import seaborn as sns
import emcee

import matplotlib
# import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

from statsmodels.robust.scale import mad as mad_calc
from sklearn import linear_model
import scipy.stats as stats
import scipy.misc as misc

def get_values():
    sfile = glob.glob('../data/AM_TRI/k1.7*.txt')[0]
    # sfile = glob.glob('../data/TRILEGAL/*.dat')[0]
    df = pd.read_csv(sfile, sep='\s+')

    '''This function corrects for extinction and sets the RC search range'''
    m_ks = df['Keplermag'].values
    mu = df['mu0'].values
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
    # vals = [-1.63,-1.59,-1.627,-1.626]
    # errs = [0.002,0.005,0.20,0.057]
    # noms = ['MixMod 2d', 'Mixmod 3d', 'RANSAC', 'Hawkins+17']
    # x = np.arange(2,10,2)
    # # fig, ax = plt.subplots()
    # # ax.errorbar(x,vals,yerr=errs, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    # # ax.scatter(x,vals,s=5,c='w',zorder=1000)
    # # ax.set_xticks(x)
    # # ax.set_xticklabels(noms,fontsize=10)
    # # ax.set_ylabel('Absolute Red Clump magnitude (TRILEGAL)')
    # # fig.tight_layout()
    # # plt.show()


    # sfile = glob.glob('../data/AM_TRI/k1.7*.txt')[0]
    # sfile = glob.glob('../data/AM_TRI/k1.7*.txt')[0]
    sfile = glob.glob('../data/TRILEGAL_sim/*all*.txt')[0]
    df = pd.read_csv(sfile, sep='\s+')

    # corr = [-0.5, -2.5, 15., 6., 1.4, -.5, .5]
    # df['Aks'] = 0.114*df.Av #Cardelli+1989>-
    # df['M_ks'] = df.Ks - df['m-M0'] - df.Aks
    #
    #
    # #Set selection criteria
    # df = df[df.M_ks < corr[0]]
    # df = df[df.M_ks > corr[1]]
    #
    # df = df[df.Ks < corr[2]]
    # df = df[df.Ks > corr[3]]

    # df = df[df.Mact < corr[4]]

    # df = df[df['[M/H]'] > corr[5]]
    # df = df[df['[M/H]'] < corr[6]]

    # df = df[df.stage < 5]

    '''This function corrects for extinction and sets the RC search range'''
    m_ks = df['Ks'].values
    mu = df['m-M0'].values
    Av = df['Av'].values
    M = df['Mact'].values
    labels = df['stage'].values
    Zish = df['[M/H]'].values
    logT = df['logTe'].values
    logL = df['logL'].values

    '''This function corrects for extinction and sets the RC search range'''
    # m_ks = df['Ksmag'].values
    # mu = df['mu0'].values
    # Av = df['Av'].values
    # M = df['Mass'].values
    # labels = df['label'].values
    # logT = df['logTe'].values
    # logL = df['logL'].values
    # Zish = df['M_H'].values


    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots(3,3)
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    label = ['Pre-Main Sequence', 'Main Sequence', 'Subgiant Branch', 'Red Giant Branch', 'Core Helium Burning',\
                'RR Lyrae variables', 'Cepheid Variables', 'Asymptotic Giant Branch','Supergiants']
    loc = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    for i in range(int(np.nanmax(labels))+1):
        if i != 5:
            if i != 6:
                if i!= 8:
                    ax.scatter(logT[labels==i],logL[labels==i],s=5,c=c[i],alpha=.3,label=label[i])
                    im = ax2[loc[i]].scatter(logT[labels==i],logL[labels==i],s=10,c=M[labels==i],\
                                        cmap='viridis',vmin=0.0,vmax=2.)
                    ax2[loc[i]].set_title(str(i))
                    ax2[loc[i]].invert_xaxis()

    ax.scatter(logT[labels==4],logL[labels==4],s=5,c=c[4],alpha=.5)
    ax.legend(loc='best',fancybox=True)
    ax.invert_xaxis()
    ax.set_xlabel(r"$log_{10}(T_{eff})$")
    ax.set_ylabel(r'$log_{10}(L)$')
    ax.set_title(r'HR Diagram for a TRILEGAL dataset of the $\textit{Kepler}$ field')
    ax.grid()
    ax.set_axisbelow(True)
    fig2.tight_layout()

    fig2.subplots_adjust(right=0.8)
    cbar_ax = fig2.add_axes([0.85,0.15,0.05,0.7])
    fig2.colorbar(im,cax=cbar_ax)

    fig.tight_layout()
    fig.savefig('/home/oliver/Dropbox/Papers/Midterm/Images/C4_HR.png')
    plt.show(fig)
    plt.close('all')

    sys.exit()

    # df = df[df.M_ks < corr[0]]
    # df = df[df.M_ks > corr[1]]
    df = df[df.stage == 4]
    df = df[df.logL < 3.2]
    m_ks = df['Ks'].values
    mu = df['m-M0'].values
    Av = df['Av'].values
    M = df['Mact'].values
    labels = df['stage'].values
    Zish = df['[M/H]'].values
    logT = df['logTe'].values
    logL = df['logL'].values



    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_axes([0.1,0.2,0.35,0.6])
    ax2 = fig.add_axes([0.55,0.2,0.35,0.6],sharex=ax1,sharey=ax1)

    s1 = ax1.scatter(logT[labels==4],logL[labels==4],s=10,c=M[labels==4],\
                cmap='viridis',vmin=0.0,vmax=4.)
    fig.colorbar(s1,label='Mass ($M_\odot$)',ax=ax1)
    ax1.set_xlabel(r"$log_{10}(T_{eff})$")
    ax1.set_ylabel(r"$log_{10}(L)$")
    ax1.set_title('CHeB stars coloured by Mass')
    ax1.invert_xaxis()

    cmap = plt.cm.get_cmap('coolwarm')
    s2 = ax2.scatter(logT[labels==4],logL[labels==4],s=10,c=Zish[labels==4],\
                cmap=cmap, vmin=-2, vmax=1)
    fig.colorbar(s2,label='[M/H]',extend='min',ax=ax2)
    cmap.set_under("b")
    ax2.set_xlabel("$log_{10}(T_{eff})$")
    ax2.set_ylabel("$log_{10}(L)$")
    ax2.set_title('CHeB stars coloured by [M/H]')

    fig.savefig('/home/oliver/Dropbox/Papers/Midterm/Images/C4_CHeB.png')
    plt.show()
    plt.close('all')



    sys.exit()
    x, y = get_values()
    xerr = np.abs(0.01 + np.random.normal(0, 1, len(x)) * 0.005)
    yerr = np.abs(0.1 + np.random.normal(0, 1, len(y)) * 0.05)


    start_params = np.array([-1.6, 0.2, 0.5, -2.0, 1.0])
    b = -1.6
    sigrc = 0.05
    o = 1.0
    sigo = 1.0

    sig1 = np.sqrt(sigrc**2 + xerr**2)
    sig2 = np.sqrt(sigo**2 + xerr**2)

    '''Trying out new likelihoods'''
    lrc = -0.5 * (x - b)**2 / sig1**2 - np.log(sig1)

    xn = np.abs(x)
    bn = np.abs(b)
    lbg = -np.log(xn) - np.log(sig2) - 0.5 * (np.log(xn) - bn)**2/sig2**2

    # plt.scatter(x,np.exp(lrc), c='r')
    plt.scatter(x,lbg, c='c')
    plt.show()


    sys.exit()

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

    # Plot the (true) good points.
    # plt.scatter(x[~m_bkg], y[~m_bkg], marker="o", s=22, c=post_prob[~m_bkg], cmap="Blues_r", vmin=0, vmax=1, zorder=1000)

    # Plot the true line.
    # plt.plot(x0, y0, color="k", lw=1.5)

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()

    plt.scatter(x, y, c=post_prob, cmap="Blues_r", s=4,zorder=1000)
    plt.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001)
    # ax.fill_between(fraclong, lolong, uplong,interpolate=True, facecolor='cyan')
    # plt.plot(rcx,rcy,lw=2,c='r',zorder=1001)
    # plt.axvline(rc+err,linestyle='--')
    # plt.axvline(rc-err,linestyle='--')
    plt.title('Clump luminosity: '+str(rc))
    plt.ylim([6,15])
    plt.show()
