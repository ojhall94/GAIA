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
import matplotlib.pyplot as plt
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

import scipy.stats as stats

import cPrior
import cLikelihood
import cMCMC

def get_values(US):
    files = glob.glob('../data/Saniya_RGBB/*'+US+'.*')[0]
    df = pd.read_table(files, sep='\s+', header=0, skiprows=3, error_bad_lines=False)
    df['logT'] = np.log10(df.Teff)
    df['logL'] = np.log10(df.L)
    df['logg'] = np.log10(df.g)
    df['lognumax'] = np.log10(df.numax)
    df = df.sort_values(by=['numax'])
    return df.lognumax, df.logT, df

class cModel:
    '''Models for this run.'''
    def __init__(self, _x, _y):
        self.x = _x
        self.y = _y

    def fg_x(self, p):
        b, sigb, _, _ = p

        #Calculating the likelihood in the X direction
        lnLx = -0.5 * (((b - self.x) / sigb)**2 + 2*np.log(sigb) +np.log(2*np.pi))
        return lnLx

    def bg_x(self, p):
        _, _, lambd, _ = p

        #Calculating the likelihood in the X direction
        A = lambd * (np.exp(lambd*x.max()) - np.exp(lambd*x.min()))**-1
        lnLx = np.log(A) + lambd*x
        return lnLx

    def bg(self, p):
        bg =  self.bg_x(p)
        return bg

    def fg(self, p):
        fg = self.fg_x(p)
        return fg


if __name__ == '__main__':
    plt.close('all')
####---SETTING UP DATA
    '''Under: 0.00, 0.025, 0.02, 0.04'''
    for US in ('0.00','0.025','0.02','0.04'):
        x, y, df = get_values(US)

        bins = int(np.sqrt(len(x)))

        #Plotting the data to be fit to
        fig, ax = plt.subplots(2, sharex=True)
        ax[0].scatter(x, y, s=3, label=US+' Undershoot', zorder=1000)
        ax[0].set_title('Synthetic Pop. for undershoot efficiencey of '+US)
        ax[0].set_ylabel(r"$log_{10}$($T_{eff}$ (K))")
        ax[0].legend(loc='best',fancybox=True)

        n, b = np.histogram(10**x,bins=bins)
        lognuguess = np.log10(b[np.argmax(n)])

        ax[1].hist(x, bins=bins, color ='k', histtype='step', normed=1)
        ax[1].axvline(lognuguess,c='r',label=r"$\nu_{max}$ estimate")
        ax[1].set_title(r"Histogram in $log_{10}$($\nu_{max}$)")
        ax[1].set_xlabel(r"$log_{10}$($\nu_{max}$ ($\mu$Hz))")
        fig.tight_layout()
        fig.savefig('Output/Saniya_RGBB/investigate_US_'+US+'.png')
        plt.close('all')

####---SETTING UP MCMC
        labels_mc = ["$b$", r"$\sigma(b)$", r"$\lambda$","$Q$"]
        start_params = np.array([lognuguess, 0.02, 1.8, 0.5])
        bounds = [(lognuguess-.05, lognuguess+.05,), (0.01,0.05),\
                    (1.4, 2.2), (0,1)]
        Model = cModel(x, y)
        lnprior = cPrior.Prior(bounds)
        Like = cLikelihood.Likelihood(lnprior,Model)

####---CHECKING MODELS BEFORE RUN
        #Getting the KDE of the 2D distribution
        xxyy = np.ones([len(x),2])
        xxyy[:,0] = x
        xxyy[:,1] = y
        kde = stats.gaussian_kde(xxyy.T)

        #Setting up a 2D meshgrid
        size = 200
        xx = np.linspace(x.min(),x.max(),size)
        yy = np.linspace(y.min(),y.max(),size)
        X, Y  = np.meshgrid(xx, yy)
        d = np.ones([size, size])

        #Calculating the KDE value for each point on the grid
        for idx, i in tqdm(enumerate(xx)):
            for jdx, j in enumerate(yy):
                d[jdx, idx] = kde([i,j])

        #Plotting residuals with histograms
        left, bottom, width, height = 0.1, 0.35, 0.8, 0.60
        fig = plt.figure(1, figsize=(8,8))
        sax = fig.add_axes([left, bottom, width, height])
        xax = fig.add_axes([left, 0.1, width, 0.22], sharex=sax)
        sax.xaxis.set_visible(False)
        xax.grid()
        xax.set_axisbelow(True)

        fig.suptitle('KDE of RGBB residuals to straight line polyfit, US '+US)

        sax.hist2d(xxyy[:,0],xxyy[:,1],bins=bins, cmap='Blues_r', zorder=1000)
        sax.contour(X,Y,d, cmap='copper', zorder=1001, label='Kernel Density',)
        sax.legend(loc='best',fancybox=True)

        xax2 = xax.twinx()
        xax2.scatter(x,np.exp(Model.fg_x(start_params)),c='cornflowerblue', label='RGBB Model')
        xax.scatter(x,np.exp(Model.bg_x(start_params)),c='orange', label='RGB Model')
        xax.hist(x,bins=bins,histtype='step',color='r',normed=True)

        sax.set_ylabel(r"$log_{10}(T_{eff})$")
        xax.set_xlabel(r"$log_{10}(\nu_{max})$")
        fig.savefig('Output/Saniya_RGBB/KDE_visual_'+US+'.png')
        plt.close('all')

####---RUNNING MCMC
        ntemps, nwalkers = 4, 32

        Fit = cMCMC.MCMC(start_params, Like, lnprior, 'none', ntemps, 1000, nwalkers)
        chain = Fit.run()

    ####---CONSOLIDATING RESULTS
        corner.corner(chain, bins=35,labels=labels_mc)
        plt.savefig('Output/Saniya_RGBB/corner_US_'+US+'.png')
        plt.close()

        lnK, fg_pp = Fit.log_bayes()
        mask = lnK > 1
        Fit.dump()

    ####---PLOTTING RESULTS
        print('Plotting results...')
        npa = chain.shape[1]
        res = np.zeros(npa)
        std = np.zeros(npa)
        for idx in np.arange(npa):
            res[idx] = np.median(chain[:,idx])
            std[idx] = np.std(chain[:,idx])

        #Plotting residuals with histograms
        #Plotting residuals with histograms
        left, bottom, width, height = 0.1, 0.35, 0.8, 0.60
        fig = plt.figure(1, figsize=(8,8))
        sax = fig.add_axes([left, bottom, width, height])
        xax = fig.add_axes([left, 0.1, width-0.15, 0.22], sharex=sax)
        sax.xaxis.set_visible(False)
        xax.grid()
        xax.set_axisbelow(True)

        fig.suptitle('KDE of RGBB residuals to straight line polyfit, US '+US)

        col = sax.scatter(x, y, c=fg_pp, cmap='viridis')
        fig.colorbar(col, ax=sax, label = 'RGBB membership posterior probability')

        xax2 = xax.twinx()
        xax2.scatter(x,np.exp(Model.fg_x(res)),c='cornflowerblue', label='RGBB Model')
        xax.scatter(x,np.exp(Model.bg_x(res)),c='orange', label='RGB Model')
        xax.hist(x,bins=bins,histtype='step',color='r',normed=True)

        sax.set_ylabel(r"$log_{10}(T_{eff})$")
        xax.set_xlabel(r"$log_{10}(\nu_{max})$")
        fig.savefig('Output/Saniya_RGBB/result_'+US+'.png')
        plt.close('all')




        #Plotting identified RGBB stars
        fig, ax = plt.subplots()
        ax.scatter(df.Teff[mask], df.numax[mask], c='y', s=3, label='RGBB Stars')
        ax.scatter(df.Teff[~mask], df.numax[~mask], c='g', s=3, label='RGB Stars')
        ax.legend(loc='best',fancybox=True)
        ax.text(4600,200,r"$\nu_{max}$ RGBB stddev = "+str.format('{0:.3f}',np.std(df.numax[mask]))+r"$\mu$Hz")
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.set_title(r"Identified RGBB stars in $\nu_{max}$ for Undershoot of "+US)
        ax.set_xlabel(r"$T_{eff}$ (K)")
        ax.set_ylabel(r"$\nu_{max}$($\mu$Hz)")
        fig.savefig('Output/Saniya_RGBB/comparison_'+US+'.png')
        plt.close('all')


        #Saving out the data with new labels
        df['label'] = ''
        df.label[mask] = 'RGBB'
        df.label[~mask] = 'RGB'

        out = glob.glob('../data/Saniya_RGBB/*'+US+'.*')[0]

        header = "#Generated synthetic population: 1000 stars\n\
        #M = 1.20 MSun, Undershoot ="+US+"\n\
        #Columns: Teff (K) - L (LSun) - numax (muHz) - dnu (muHz) - g (cm/s^2) - logT - logL - logg - lognumax - label\n\
        #Teff\t\t\t\tL\t\t\t\tnumax\t\t\t\tdnu\t\t\t\tg\t\t\t\tlogT\t\t\t\tlogL\t\t\t\tlogg\t\t\t\tlognumax\t\t\t\tlabel"
        df.to_csv('../data/Saniya_RGBB/m1.20.ovh0.01d.ovhe0.50s.z0.01756.y0.26618.under'+US+'_labeled.txt',header=header,sep='\t')
