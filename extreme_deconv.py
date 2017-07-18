#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# JSK

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
from xdgmm import XDGMM

from sklearn.model_selection import validation_curve
from sklearn.model_selection import ShuffleSplit
from astroML.plotting.tools import draw_ellipse
from astroML.plotting import setup_text_plots
from sklearn.mixture import GMM as skl_GMM
import warnings
warnings.filterwarnings('ignore')

def plot_sample(x_true, y_true, x, y, sample, xdgmm):
    setup_text_plots(fontsize=16, usetex=True)
    plt.clf()
    fig = plt.figure(figsize=(12, 9))
    fig.subplots_adjust(left=0.1, right=0.95,
                        bottom=0.1, top=0.95,
                        wspace=0.02, hspace=0.02)

    ax1 = fig.add_subplot(221)
    ax1.scatter(x_true, y_true, s=4, lw=0, c='k')

    ax2 = fig.add_subplot(222)

    ax2.scatter(x, y, s=4, lw=0, c='k')

    ax3 = fig.add_subplot(223)
    ax3.scatter(sample[:, 0], sample[:, 1], s=4, lw=0, c='k')

    ax4 = fig.add_subplot(224)
    for i in range(xdgmm.n_components):
        draw_ellipse(xdgmm.mu[i], xdgmm.V[i], scales=[2], ax=ax4,
                     ec='k', fc='gray', alpha=0.2)

    titles = ["True Distribution", "Noisy Distribution",
              "Extreme Deconvolution\n  resampling",
            "Extreme Deconvolution\n  cluster locations"]

    ax = [ax1, ax2, ax3, ax4]

    for i in range(4):
        ax[i].set_xlim(-2.1, -0.2)
        ax[i].set_ylim(6.5, 14.5)

        #ax[i].xaxis.set_major_locator(plt.MultipleLocator(4))
        #ax[i].yaxis.set_major_locator(plt.MultipleLocator(5))

        ax[i].text(0.05, 0.95, titles[i],
                   ha='left', va='top', transform=ax[i].transAxes)

        if i in (0, 1):
            ax[i].xaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax[i].set_xlabel('$x$', fontsize = 18)

        if i in (1, 3):
            ax[i].yaxis.set_major_formatter(plt.NullFormatter())
        else:
            ax[i].set_ylabel('$y$', fontsize = 18)
    plt.savefig('Extreme_Deconvolution_test.png')
    plt.show()


def plot_bic(param_range,bics,lowest_comp):
    plt.clf()
    setup_text_plots(fontsize=16, usetex=True)
    fig = plt.figure(figsize=(12, 6))
    plt.bar(param_range-0.25,bics,color='blue',width=0.5)
    plt.text(lowest_comp, bics.min() * 0.97 + .03 * bics.max(), '*',
             fontsize=14, ha='center')

    plt.xticks(param_range)
    plt.ylim(bics.min() - 0.01 * (bics.max() - bics.min()),
             bics.max() + 0.01 * (bics.max() - bics.min()))
    plt.xlim(param_range.min() - 1, param_range.max() + 1)

    plt.xticks(param_range,fontsize=14)
    plt.yticks(fontsize=14)


    plt.xlabel('Number of components',fontsize=18)
    plt.ylabel('BIC score',fontsize=18)

    plt.show()

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

if __name__ == '__main__':


    M_ks, m_ks = np.loadtxt('data.txt', unpack=True).T
    #
    x = M_ks[::5]
    y = m_ks[::5]


    # Make up some uncertainties on magnitudes
    dx = 0.05 + np.random.normal(0, 1, len(x)) * 0.005
    dy = 0.1 + np.random.normal(0, 1, len(y)) * 0.05

    # stack the results for computation
    X = np.vstack([x, y]).T
    Xerr = np.zeros(X.shape + X.shape[-1:])
    diag = np.arange(X.shape[-1])
    Xerr[:, diag, diag] = np.vstack([dx ** 2, dy ** 2]).T

    plt.errorbar(x, y, xerr=dx, yerr=dy, fmt='.')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()

    # Instantiate an XDGMM model:
    xdgmm = XDGMM(method='Bovy')

    # Define the range of component numbers, and get ready to compute the BIC for each one:
    param_range = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

    # Loop over component numbers, fitting XDGMM model and computing the BIC:
    bic, optimal_n_comp, lowest_bic = xdgmm.bic_test(X, Xerr, param_range)

    plot_bic(param_range, bic, optimal_n_comp)

    xdgmm.n_components = optimal_n_comp
    xdgmm = xdgmm.fit(X, Xerr)
    N = len(x)
    sample = xdgmm.sample(N)

    print(xdgmm.GMM.alpha)
    print(xdgmm.GMM.mu)
    print(xdgmm.GMM.V)
    plot_sample(x, y, x, y, sample, xdgmm)
