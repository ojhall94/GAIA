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

    def fg(self, p):
        b, sigb, _, _, _, _ = p
        return -0.5 * (((b - self.x) / sigb)**2 + 2*np.log(sigb) + np.log(2*np.pi))

    def bg(self, p):
        _, _, m, c, sigm, _ = p
        model = m * x + c
        return -0.5 * (((model - self.y) / sigm)**2 + 2*np.log(sigm) + np.log(2*np.pi))
    #
    # def bg(self, p):
    #     _, _, o, sigo, _ = p
    #     val = np.abs(self.x).max() + np.abs(self.x).min()
    #
    #     xn = self.x + val
    #     on = np.abs(o)
    #
    #     return -np.log(xn) -np.log(sigo) - 0.5 * (np.log(xn) - on)**2/sigo**2




if __name__ == '__main__':
    plt.close('all')
####---SETTING UP DATA
    '''Under: 0.00, 0.025, 0.02, 0.04'''
    for US in ('0.00','0.025','0.02','0.04'):
        x, y, df = get_values(US)
        print('WARNING: ALL DATA IN LOG BASE 10')

        '''Estimate background parameters in log space'''
        f = np.polyfit(x, y, 1)
        bins = int(np.sqrt(len(x)))

        fig, ax = plt.subplots(2, sharex=True)
        ax[0].scatter(x, y, s=3, label=US+' Undershoot', zorder=1000)
        ax[0].plot(x,f[1] + x*f[0],c='r', label='Polyfit', zorder=999)
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
        plt.close()


    ####---SETTING UP AND RUNNING MCMC
        # labels_mc = ["$b$", r"$\sigma(RC)$", "$o$", r"$\sigma(o)$", "$Q$"]
        # bounds = [(lognuguess-.05, lognuguess+.05,), (0.01,0.05), (0.0,2.0), (0.01, 2.), (0, 1)]
        # start_params = np.array([lognuguess, 0.02, 0.1, 1.0, 0.5])
        labels_mc = ["$b$", r"$\sigma(b)$", "$m$", "$c$", r"$\sigma(m)$", "$Q$"]
        std = np.std(y-(x*f[0]+f[1]))
        start_params = np.array([lognuguess, 0.02, f[0], f[1], std, 0.5])
        bounds = [(lognuguess-.05, lognuguess+.05,), (0.01,0.05),\
                     (f[0]-0.02,f[0]+0.02), (f[1]-0.3,f[1]+0.3), \
                    (std/5,std*5), (0, 1)]


        Model = cModel(x, y)
        lnprior = cPrior.Prior(bounds)
        Like = cLikelihood.Likelihood(lnprior,Model)

        ntemps, nwalkers = 2, 32

        Fit = cMCMC.MCMC(start_params, Like, lnprior, 'none', ntemps, 500, nwalkers)
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

        fig, ax = plt.subplots(2, sharex=True)
        ax[0].scatter(x, y, s=5, c=fg_pp, cmap='Blues_r', vmin=0, vmax=1, zorder=1000)
        # ax[0].plot(x,res[2]+res[3]*x,c='r',zorder=1001,label='MCMC BG Fit')
        ax[0].set_title('Synthetic Pop. for undershoot efficiencey of '+US)
        ax[0].set_ylabel(r"$log_{10}$($T_{eff}$ (K))")
        ax[0].legend(loc='best',fancybox=True)
        ax[0].axvline(res[0],c='r',label=r"$RGBB location")

        fg_m = np.exp(Model.fg(res))
        bg_m = np.exp(Model.bg(res))
        weights = np.ones_like(x)/float(len(x))
        hy, _, _ = ax[1].hist(x, weights = weights, bins=bins, color ='k', histtype='step')
        ax[1].scatter(x,fg_m/fg_m.max()*hy.max(),c='cornflowerblue',alpha=.5,label='FG',s=5)
        ax[1].scatter(x,bg_m/fg_m.max()*hy.max(),c='orange',alpha=.5,label='BG',s=5)
        ax[1].axvline(res[0],c='r',label=r"$RGBB location")
        ax[1].set_title(r"Histogram in $log_{10}$($\nu_{max}$)")
        ax[1].set_xlabel(r"$log_{10}$($\nu_{max}$ ($\mu$Hz))")
        fig.tight_layout()
        fig.savefig('Output/Saniya_RGBB/results_'+US+'.png')
        plt.close()

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

        sys.exit()

        df['label'] = ''
        df.label[mask] = 'RGBB'
        df.label[~mask] = 'RGB'

        out = glob.glob('../data/Saniya_RGBB/*'+US+'.*')[0]

        header = "#Generated synthetic population: 1000 stars\n\
        #M = 1.20 MSun, Undershoot ="+US+"\n\
        #Columns: Teff (K) - L (LSun) - numax (muHz) - dnu (muHz) - g (cm/s^2) - logT - logL - logg - lognumax - label\n\
        #Teff\t\t\t\tL\t\t\t\tnumax\t\t\t\tdnu\t\t\t\tg\t\t\t\tlogT\t\t\t\tlogL\t\t\t\tlogg\t\t\t\tlognumax\t\t\t\tlabel"
        df.to_csv('../data/Saniya_RGBB/m1.20.ovh0.01d.ovhe0.50s.z0.01756.y0.26618.under'+US+'_labeled.txt',header=header,sep='\t')
