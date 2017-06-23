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
import seaborn as sns
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

    sel = np.where(m_ks < 10.)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]

    sel = np.where(m_ks > 7.)
    M_ks = M_ks[sel]
    m_ks = m_ks[sel]

    return M_ks, m_ks

def linear(x,y,X):
    #Fit RANSAC linear regressor
    #Use Hawkins+17 error: 0.057

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression(),\
                                                residual_threshold=(0.057))
    model_ransac.fit(Y,x)
    inlier_mask = model_ransac.inlier_mask_
    line_Y = np.arange(0,len(Y))

    line_x = model_ransac.predict(line_Y[:,np.newaxis])

    inlier_mask = model_ransac.inlier_mask_
    coef = np.ones(2)
    coef[1] = model_ransac.estimator_.coef_.flatten()
    coef[0] = model_ransac.estimator_.intercept_

    return coef, inlier_mask

def fBounds():
    start = [0.1,0.1,0.5,0.2,0.1]
    bound = [(0.,0.3),(0.01,0.2),(0.,1.),(0.,1.),(0.01,0.2)]
    return start,bound

class cModel:
    def __init__(self, _X, _Y, _data):
        self.X = _X
        self.Y = _Y
        self.data = _data

    def bg(self, p):
        c,_,_,_,_ = p
        return np.ones_like(self.X) * c

class cLikelihood:
    def __init__(self, _data, _Model, _Prior):
        self.data = _data
        self.Model = _Model
        self.Prior = _Prior

    def lnlike_bg(self, p):
        _,b_var,Q,_,_ = p
        mod = self.Model.bg(p)
        ll_bg = -0.5 * (self.data - mod)**2 / b_var**2 - np.log(b_var)
        arg1 = ll_bg + np.log(1.0 - Q)
        return arg1

    def lnlike_fg(self, p):
        _,_,Q,M,f_var = p
        ll_fg = -0.5 * (self.data - M)**2 / f_var**2 - np.log(f_var)
        arg2 = ll_fg + np.log(Q)
        return arg2

    def lnprob(self, p):
        if not np.isfinite(self.Prior(p)):
            return -np.inf

        if p[0] > p[3]:
            return -np.inf

        #Doing the likelihood equation in parts
        arg1 = self.lnlike_fg(p)
        arg2 = self.lnlike_bg(p)

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
        ax1[0].set_ylim(7.0,10.0)

        #_________________Building Density distribution_________________________
        x = M_ks
        y = m_ks

        #Getting the KDE of the 2D distribution
        xxyy = np.ones([len(x),2])
        xxyy[:,0] = x
        xxyy[:,1] = y
        kde = stats.gaussian_kde(xxyy.T)

        #Setting up a 2D meshgrid
        size = 50
        xx = np.linspace(x.min(),x.max(),size)
        yy = np.linspace(y.min(),y.max(),size)
        X, Y = np.meshgrid(xx,yy)
        d = np.ones([size,size])

        #Calculating the KDE value for each point on the grid
        for idx, i in tqdm(enumerate(xx)):
            for jdx, j in enumerate(yy):
                d[jdx,idx] = kde([i,j])

        #Rough attempt at finding line of highest density
        line = np.zeros_like(yy)
        for idy, row in enumerate(yy):
            space = d[idy]
            med = np.nanmax(space)
            line[idy] = xx[space==med]

        #Plotting density with estimated Absolute RC Magnitude
        ax1[1].hist2d(xxyy[:,0],xxyy[:,1],bins=np.sqrt(len(x)))
        ax1[1].contour(X,Y,d,10,cmap='copper')
        ax1[1].plot(line,yy,c='r')
        ax1[1].set_xlabel('Absolute K-band magnitude')
        ax1[1].set_ylabel('Apparent K-band magnitude')
        ax1[1].axvline(-1.626,c='k',linestyle='--') # Hawkins+17
        ax1[1].axvline(-1.626+0.057, c='r', linestyle='-.')
        ax1[1].axvline(-1.626-0.057, c='r', linestyle='-.')
        ax1[1].set_xlim(-2.0,-0.25)
        ax1[1].set_ylim(7.0,10.0)
        fig1.tight_layout()

        #Plotting density in 3D for clairty
        fig2 = plt.figure()
        ax2 = fig2.gca(projection='3d')
        ax2.scatter(X,Y,d/d.max(),c=d/d.max(),cmap='Blues')
        ax2.set_xlabel('Absolute Magnitude')
        ax2.set_ylabel('Apparent Magnitude')
        ax2.set_zlabel('Normalised Kernel Estimated Density')

        plt.show()
        plt.close('all')

        #_________________Building Density distribution_________________________

    if fold:
        print('hello world')

    start_params, bounds = fBounds()
    Prior = cPrior.Prior(_bounds = bounds)

    Model = cModel(X,Y,d)
    Like = cLikelihood(d,Model,Prior)

    print "\nInitialising run"
    print "\nLike ##########", Like(start_params)
    print "Prior #########", Prior(start_params), "\n"
    print "Starting guesses:\n", start_params
    print "Priors: \n", bounds

    Fit = cMCMC.MCMC('TRILEGAL',start_params, Like, Prior,\
                    _start_kdes=0,_ntemps=1,_niter=500)

    chain = Fit.run()
    fg_pp,_ = Fit.postprob(X)

    labels=['c','c_var','Q','f_M','f_var']
    figc = corner.corner(chain, labels=labels)

    results = pd.DataFrame(columns=['a','b','c','Q','w'])
    stddevs = pd.DataFrame(columns=['aerr','berr','cerr','Qerr','werr'])
    results.loc[0], stddevs.loc[0] = fResults(chain)

    print(chain.shape)
    print(results.loc[0])
    print(stddevs.loc[0])

    #Plots 3d scatter
    fig1 = plt.figure()
    ax1 = fig1.gca(projection='3d')
    scat1 = ax1.scatter(X,Y,d,c=fg_pp,cmap="Blues_r")
    cmap = fig1.colorbar(scat1,label='Posterior Probability')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Normalised Kernel Estimated Density')

    #Plots top down pixel view (works better on larger arrays)
    fig2, ax2 = plt.subplots()
    scat2 = ax2.scatter(X,Y,marker=",",c=fg_pp,s=200,cmap="Blues_r")
    cmap = fig2.colorbar(scat2,label='Posterior Probability')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')

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
