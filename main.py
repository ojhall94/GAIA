# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Oliver J. Hall

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import glob
import sys
import matplotlib
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

from statsmodels.robust.scale import mad as mad_calc
from sklearn import linear_model
import corner as corner


import astropy.coordinates as coord
import gaia.tap as gt

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

    # BIC = get_BIC(x,y,line_y,inlier_mask,coef)

    return coef, inlier_mask


if __name__ == '__main__':
    trilegal = True
    if trilegal:
        sfile = glob.glob('../data/TRILEGAL_sim/*.all.*.txt')[0]
        # sfile = glob.glob('../data/TRILEGAL/*.dat')[0]
        df = pd.read_csv(sfile, sep='\s+')

        M_ks, m_ks = get_values(df)
        fig, ax = plt.subplots(2)
        ax[0].scatter(M_ks, m_ks, edgecolor='b',facecolor='none')
        ax[0].set_xlabel('Absolute K-band magnitude')
        ax[0].set_ylabel('Apparent K-band magnitude')
        ax[0].axvline(-1.626,c='k',linestyle='--') # Hawkins+17
        ax[0].axvline(-1.626+0.057, c='r', linestyle='-.')
        ax[0].axvline(-1.626-0.057, c='r', linestyle='-.')
        ax[0].set_xlim(-2.0,-0.25)
        ax[0].set_ylim(7.0,10.0)

        '''Time for a RANSAC plot'''
        iters = 500
        x = M_ks
        y = m_ks
        Y = y.reshape((len(y),1))
        line_Y = np.arange(0,len(Y))

        #Setting up the array
        sample = np.ones([iters,2])
        mask = np.zeros(len(y))

        for i in range(iters):
            sample[i], inlier_mask = linear(x,y,Y)
            mask += np.ones(len(inlier_mask)) * inlier_mask

        labels = ['Intercept','Slope']
        figc = corner.corner(sample, labels=labels)

        results = np.zeros(2)
        results[0] = np.median(sample[:,0])
        results[1] = np.median(sample[:,1])

        ly = np.linspace(y.min(),y.max(),100)
        lx = results[0] + results[1]*ly

        mask = mask > (iters/2)

        # ax[1].scatter(M_ks[mask],m_ks[mask],s=5,color='yellowgreen',label='Inliers')
        # ax[1].scatter(M_ks[~mask],m_ks[~mask],s=5,color='grey',label='Outliers')
        ax[1].hist2d(x,y,bins=np.sqrt(len(x)))
        ax[1].plot(lx,ly,c='red',lw=2,label='RANSAC fit')

        ax[1].set_xlabel('Absolute K-band magnitude')
        ax[1].set_ylabel('Apparent K-band magnitude')
        ax[1].axvline(-1.626,c='y',lw=3,linestyle='--',label='Hawkins+17') # Hawkins+17
        ax[1].axvline(-1.626+0.057, c='y', lw=2,linestyle='-.')
        ax[1].axvline(-1.626-0.057, c='y', lw=2,linestyle='-.')
        ax[1].set_xlim(-2.0,-0.25)
        ax[1].set_ylim(7.0,10.0)
        ax[1].legend()

        fig.tight_layout()
        fig.savefig('ransac.png')
        figc.savefig('ransac_corner.png')

        plt.show('all')
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
