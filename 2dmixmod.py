# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Oliver J. Hall

import numpy as np
import pandas as pd

import glob
import sys
import corner as corner
from tqdm import tqdm

import matplotlib
# import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

from statsmodels.robust.scale import mad as mad_calc
from sklearn import linear_model
import scipy.stats as stats

import astropy.coordinates as coord
import gaia.tap as gt

import cMCMC
import cPrior

def get_values(df):
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

    return M_ks, m_ks


class cModel:
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    def fg(self, p):
        m,Q,M,V = p
        model = np.ones(len(self.x)) * m
        return model

    '''Background model is generic'''

class cLikelihood:
    def __init__(self, _x, _xerr, _y, _Model, _Prior):
        self.x = _x
        self.y = _y
        self.xerr = _xerr
        self.Model = _Model
        self.Prior = _Prior

    def lnlike_fg(self, p):
        m,Q,M,V = p
        mod = self.Model.fg(p)

        ll_fg = -0.5 * (mod - x)**2/self.xerr**2 - np.log(xerr)
        arg2 = ll_fg + np.log(Q)
        return arg2

    def lnlike_bg(self, p):
        m,Q,M,V = p

        ll_bg = -0.5 * (M - y)**2/V**2 - np.log(V)
        arg1 = ll_bg + np.log(1.0 - Q)
        return arg1

    def lnprob(self, p):
        if not np.isfinite(self.Prior(p)):
            return -np.inf

        #Doing the likelihood equation in parts
        arg2 = self.lnlike_fg(p)
        arg1 = self.lnlike_bg(p)

        ll = np.nansum(np.logaddexp(arg1, arg2))

        return ll

    def __call__(self, p):
        logL = self.lnprob(p)
        return logL

def fResults(chain):
    '''Returns standard MCMC results as the median of the solutions, with
    errors as the standard deviation on the solutions.
    '''
    npa = chain.shape[1]
    results = np.zeros(npa)
    stddevs = np.zeros(npa)
    for i in np.arange(npa):
        results[i] = np.median(chain[:,i])
        stddevs[i] = np.std(chain[:,i])

    return results, stddevs


if __name__ == '__main__':
    fold = True
    '''KDE build'''
    if fold:
        sfile = glob.glob('../data/TRILEGAL_sim/*.all.*.txt')[0]
        # sfile = glob.glob('../data/TRILEGAL/*.dat')[0]
        df = pd.read_csv(sfile, sep='\s+')

        #Plotting data with RC Absolute magnitude
        M_ks, m_ks = get_values(df)

        fig1, ax1 = plt.subplots(2)
        ax1[0].scatter(M_ks, m_ks, edgecolor='b',facecolor='none')
        ax1[0].set_xlabel('Absolute K-band magnitude')
        ax1[0].set_ylabel('Apparent K-band magnitude')
        ax1[0].axvline(-1.626,c='k',linestyle='--') # Hawkins+17
        ax1[0].axvline(-1.626+0.057, c='r', linestyle='-.')
        ax1[0].axvline(-1.626-0.057, c='r', linestyle='-.')
        ax1[0].set_xlim(-2.0,-0.25)
        ax1[0].set_ylim(7.0,16.0)

        #_________________Building Density distribution_________________________
        x = M_ks
        xerr = 0.1 *  x
        y = m_ks


        #Getting the KDE of the 2D distribution
        xxyy = np.ones([len(x),2])
        xxyy[:,0] = x
        xxyy[:,1] = y

        #Plotting density with estimated Absolute RC Magnitude
        ax1[1].hist2d(xxyy[:,0],xxyy[:,1],bins=np.sqrt(len(x)))
        ax1[1].set_xlabel('Absolute K-band magnitude')
        ax1[1].set_ylabel('Apparent K-band magnitude')
        ax1[1].axvline(-1.626,c='k',linestyle='--') # Hawkins+17
        ax1[1].axvline(-1.626+0.057, c='r', linestyle='-.')
        ax1[1].axvline(-1.626-0.057, c='r', linestyle='-.')
        ax1[1].set_xlim(-2.0,-0.25)
        ax1[1].set_ylim(7.0,16.0)
        fig1.tight_layout()

        plt.show()
        plt.close('all')

        #_________________Building Density distribution_________________________

    '''MCMC Run'''
    if fold:
        '''RC_M | Q | Gauss M | V'''
        start_params = [-1.5, 0.5, 12.,1.0]
        bounds = [(-2.0,-0.25), (0,1.),\
                (7.0,15.0),(0.01,8.)]

        Prior = cPrior.Prior(_bounds = bounds)

        Model = cModel(x,y)
        Like = cLikelihood(x,xerr,y,Model,Prior)

        print "\nInitialising run"
        print "\nLike ##########", Like(start_params)
        print "Prior #########", Prior(start_params), "\n"
        print "Starting guesses:\n", start_params
        print "Priors: \n", bounds

        Fit = cMCMC.MCMC('TRILEGAL',start_params, Like, Prior,\
                        _start_kdes=0,_niter=100)

        chain = Fit.run()
        fg_pp,_ = Fit.postprob(x)

        labels=['$RC_M$','$Q$','$M$','$LnV$']
        figc = corner.corner(chain, labels=labels)

        results = pd.DataFrame(columns=['m','Q','M','LnV'])
        stddevs = pd.DataFrame(columns=['m_err','Q_err','M_err','V_err'])
        results.loc[0], stddevs.loc[0] = fResults(chain)

        print(chain.shape)
        print(results.loc[0])
        print(stddevs.loc[0])

        fg = Model.fg(results.loc[0])

        fig, ax = plt.subplots()
        ax.scatter(M_ks, m_ks,c=fg_pp)
        ax.plot(fg,y,c='c',linewidth=3) #Best fit model
        ax.plot(x,results.loc[0]['M']*np.ones_like(x),c='r',linewidth=3)
        ax.set_xlabel('Absolute K-band magnitude')
        ax.set_ylabel('Apparent K-band magnitude')
        ax.axvline(-1.626,c='k',linestyle='--') # Hawkins+17
        ax.axvline(-1.626+0.057, c='r', linestyle='-.')
        ax.axvline(-1.626-0.057, c='r', linestyle='-.')
        ax.set_xlim(-2.0,-0.25)
        ax.set_ylim(7.0,16.0)
        plt.show()
        plt.close('all')

    gaia_on_tap = False
    if gaia_on_tap:
        '''
        Kepler FOV is 105 square degrees, so sqrt(115)/2 in each direction from the centre.
        RA: 19h 22m 40s / 19.378
        Dec: +44 30' 00'' / +44.50
        '''

        '''
        Need the absolute-apparent magnitude conversion
        Need to see whether this gaia data is representative
        '''

        ra = (19. + 22./60. + 40./3600.)*15.
        dec = 44. + 30./60.
        r = np.sqrt(105)/2

        sources = gt.cone_search(ra, dec, r, table="gaiadr1.tgas_source")

        print(sources)

        fig, ax = plt.subplots()
        ax.scatter(sources['ra'], sources['dec'],s=1,c="#000000")
        ax.set_xlabel(r"$\alpha$")
        ax.set_ylabel(r"$\delta$")
        plt.show()
