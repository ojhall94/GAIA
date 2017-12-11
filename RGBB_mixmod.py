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

    def prob_y(self, p):
        _, _, m, c, sigm, _, _ = p
        #Calculating the likelihood in the Y direction
        model = m * self.x + c
        lnLy = -0.5 * (((y - model) / sigm)**2 + 2*np.log(sigm) +np.log(2*np.pi))
        return lnLy

    def fg_x(self, p):
        b, sigb, _, _, _, _, _ = p

        #Calculating the likelihood in the X direction
        lnLx = -0.5 * (((b - self.x) / sigb)**2 + 2*np.log(sigb) +np.log(2*np.pi))
        return lnLx

    def bg_x(self, p):
        _, _, _, _, _, lambd, _ = p

        #Calculating the likelihood in the X direction
        A = lambd * (np.exp(lambd*x.max()) - np.exp(lambd*x.min()))**-1
        lnLx = np.log(A) + lambd*x
        return lnLx

    def bg(self, p):
        bg =  self.prob_y(p) + self.bg_x(p)
        return bg

    def fg(self, p):
        fg = self.prob_y(p) + self.fg_x(p)
        return fg


if __name__ == '__main__':
    plt.close('all')
####---SETTING UP DATA
    '''Under: 0.00, 0.025, 0.02, 0.04'''
    for US in ('0.00','0.025','0.02','0.04'):
        x, y, df = get_values(US)

        #Estimate background parameters in log space
        f = np.polyfit(x, y, 1)
        fn = f[1] + x*f[0]
        bins = int(np.sqrt(len(x)))

        #Plotting the data to be fit to
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
        plt.close('all')

        #Getting distribution visualisation with KDE
        fn = f[1] + x*f[0]
        D = y - fn


####---SETTING UP MCMC
        labels_mc = ["$b$", r"$\sigma(b)$", "$m$", "$c$", r"$\sigma(m)$", r"$\lambda$","$Q$"]
        std = np.std(D)
        start_params = np.array([lognuguess, 0.02, f[0], f[1], std, 1.8, 0.5])
        bounds = [(lognuguess-.05, lognuguess+.05,), (0.01,0.03),\
                    (f[0]-0.02,f[0]+0.02), (f[1]-0.3,f[1]+0.3), \
                    (std*0.5,std*1.5), (1.4, 2.2), (0,1)]
        Model = cModel(x, y)
        lnprior = cPrior.Prior(bounds)
        Like = cLikelihood.Likelihood(lnprior,Model)

####---CHECKING MODELS BEFORE RUN
        #Getting the KDE of the 2D distribution
        Dxxyy = np.ones([len(x),2])
        Dxxyy[:,0] = x
        Dxxyy[:,1] = D
        Dkde = stats.gaussian_kde(Dxxyy.T)

        #Setting up a 2D meshgrid
        size = 200
        Dxx = np.linspace(x.min(),x.max(),size)
        Dyy = np.linspace(D.min(),D.max(),size)
        DX, DY  = np.meshgrid(Dxx, Dyy)
        Dd = np.ones([size, size])

        #Calculating the KDE value for each point on the grid
        for idx, i in tqdm(enumerate(Dxx)):
            for jdx, j in enumerate(Dyy):
                Dd[jdx, idx] = Dkde([i,j])

        #Plotting residuals with histograms
        left, bottom, width, height = 0.1, 0.35, 0.65, 0.60
        fig = plt.figure(1, figsize=(10,10))
        sax = fig.add_axes([left, bottom, width, height])
        yax = fig.add_axes([left+width+0.02, bottom, 0.2, height])
        xax = fig.add_axes([left, 0.1, width, 0.22], sharex=sax)
        sax.xaxis.set_visible(False)
        yax.set_yticklabels([])
        xax.grid()
        xax.set_axisbelow(True)
        yax.grid()
        yax.set_axisbelow(True)

        fig.suptitle('KDE of RGBB residuals to straight line polyfit, US '+US)

        sax.hist2d(Dxxyy[:,0],Dxxyy[:,1],bins=np.sqrt(len(x)))
        sax.contour(DX,DY,Dd)
        sax.axhline(0.,c='r',label='Polyfit',zorder=1001)
        sax.legend(loc='best',fancybox=True)

        yax.hist(D,bins=int(np.sqrt(len(D))),histtype='step',orientation='horizontal', normed=True)
        yax.scatter(np.exp(Model.prob_y(start_params)), D,c='orange')
        yax.set_ylim(sax.get_ylim())

        xax.hist(x,bins=int(np.sqrt(len(x))),histtype='step',normed=True)
        xax.scatter(x,np.exp(Model.bg_x(start_params)),c='orange')
        xax.scatter(x,np.exp(Model.fg_x(start_params)),c='cornflowerblue')

        sax.set_ylabel(r"$log_{10}(T_{eff})$ - Straight Line Model")
        xax.set_xlabel(r"$log_{10}(\nu_{max})$")
        fig.savefig('Output/Saniya_RGBB/KDE_visual_'+US+'.png')
        plt.show()
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
        left, bottom, width, height = 0.1, 0.35, 0.65, 0.60
        fig = plt.figure(1, figsize=(10,10))
        sax = fig.add_axes([left, bottom, width, height])
        yax = fig.add_axes([left+width+0.02, bottom, 0.2, height])
        xax = fig.add_axes([left, 0.1, width, 0.22], sharex=sax)
        sax.xaxis.set_visible(False)
        yax.set_yticklabels([])
        xax.grid()
        xax.set_axisbelow(True)
        yax.grid()
        yax.set_axisbelow(True)

        fig.suptitle('Resulting probability distributions, US '+US)

        fn = res[2]*x + res[3]
        Dr = y - fn
        sax.scatter(x, Dr, c=fg_pp,cmap='Blues_r',label=US)
        sax.axhline(0.,c='r',label='Straight line fit',zorder=1001)
        sax.legend(loc='best',fancybox=True)

        yax.hist(Dr,bins=int(np.sqrt(len(Dr))),histtype='step',orientation='horizontal', normed=True)
        yax.scatter(np.exp(Model.prob_y(res)), Dr,c='orange')
        yax.set_ylim(sax.get_ylim())

        xax.hist(x,bins=int(np.sqrt(len(x))),histtype='step',normed=True)
        xax.scatter(x,np.exp(Model.bg_x(res)),c='orange')
        xax.scatter(x,np.exp(Model.fg_x(res)),c='cornflowerblue')

        sax.set_ylabel(r"$log_{10}(T_{eff})$ - Straight Line Model")
        xax.set_xlabel(r"$log_{10}(\nu_{max})$")
        fig.savefig('Output/Saniya_RGBB/result_'+US+'.png')
        plt.show()
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

        sys.exit()
        out = glob.glob('../data/Saniya_RGBB/*'+US+'.*')[0]

        header = "#Generated synthetic population: 1000 stars\n\
        #M = 1.20 MSun, Undershoot ="+US+"\n\
        #Columns: Teff (K) - L (LSun) - numax (muHz) - dnu (muHz) - g (cm/s^2) - logT - logL - logg - lognumax - label\n\
        #Teff\t\t\t\tL\t\t\t\tnumax\t\t\t\tdnu\t\t\t\tg\t\t\t\tlogT\t\t\t\tlogL\t\t\t\tlogg\t\t\t\tlognumax\t\t\t\tlabel"
        df.to_csv('../data/Saniya_RGBB/m1.20.ovh0.01d.ovhe0.50s.z0.01756.y0.26618.under'+US+'_labeled.txt',header=header,sep='\t')
