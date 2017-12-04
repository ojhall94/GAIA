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
    return df.lognumax, df.Teff, df

class cModel:
    '''Models for this run.'''
    def __init__(self, _x, _y, _amp_offset, _a):
        self.x = _x
        self.y = _y
        self.amp_offset = _amp_offset
        self.a = _a

    def fg(self, p):
        b, sigb, _, _, _, _ = p
        return -0.5 * (((b - self.x) / sigb)**2 + 2*np.log(sigb))

    def bg(self, p):
        _, _, m, c, sigm, _ = p
        model = m * self.x + c
        Delta = y - model
        A = self.a * self.x + self.amp_offset
        return -0.5 * (((Delta) / sigm)**2 + 2*np.log(sigm)) + np.log(A)


if __name__ == '__main__':
    plt.close('all')
####---SETTING UP DATA
    '''Under: 0.00, 0.025, 0.02, 0.04'''
    for US in ('0.00','0.025','0.02','0.04'):
        x, y, df = get_values(US)
        print('WARNING: ALL DATA IN LOG BASE 10')

        '''Estimate background parameters in log space'''
        f = np.polyfit(x, y, 1)
        fn = f[1] + x*f[0]
        bins = int(np.sqrt(len(x)))

        '''Plotting the data to be fit to'''
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
        plt.show()
        fig.savefig('Output/Saniya_RGBB/investigate_US_'+US+'.png')
        plt.close('all')

        '''Getting distribution visualisation with KDE'''
        fn = f[1] + x*f[0]
        D = y - fn

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

        '''Plotting residuals with histograms'''
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

        yax.hist(D,bins=int(np.sqrt(len(D))),histtype='step',orientation='horizontal')
        yax.set_ylim(sax.get_ylim())
        xax.hist(x,bins=int(np.sqrt(len(x))),histtype='step')

        sax.set_ylabel(r"$T_{eff}$ - Straight Line Model")
        xax.set_xlabel(r"$log_{10}()\nu_{max})$")
        fig.savefig('Output/Saniya_RGBB/KDE_visual_'+US+'.png')

        plt.show()


        plt.show()
        sys.exit()



        '''Plotting and calculating the change in distribution amplitude'''
        diff = y - fn
        scope = np.arange(1., 2.75, 0.25)
        fig, ax = plt.subplots(2)
        lo = 0.75
        i = 0
        nums = np.ones_like(scope)
        xes = np.ones_like(scope)
        for hi in scope:
            data1 = diff[x > lo]
            data = data1[x < hi]
            n, _,_ = ax[0].hist(data,bins=int(np.sqrt(len(data))),histtype='step',label=str(lo)+'-'+str(hi))
            nums[i] = n.max()
            xes[i] = np.mean([lo,hi])
            i += 1
            lo = hi
        ax[0].legend(loc='best',fancybox=True)
        ax[0].set_xlabel(r"$T_{eff}$ - Polyfit")
        ax[0].set_ylabel('Counts per bin')

        l = np.polyfit(xes, nums, 1)
        ax[1].scatter(xes,nums)
        ax[1].plot(xes,xes*l[0]+l[1],label='Polyfit')
        ax[1].plot(xes,xes*l[0]*1.5+l[1],label='Upper Limit')
        ax[1].plot(xes,xes*l[0]*0.5+l[1], label='Lower Limit')
        ax[1].legend(loc='best',fancybox=True)
        ax[1].set_xlabel(r"$\nu_{max}$")
        ax[1].set_ylabel(r"Distribution Amplitude")
        fig.tight_layout()
        fig.savefig('Output/Saniya_RGBB/amplitude_US_'+US+'.png')
        plt.close('all')

####---SETTING UP AND RUNNING MCMC
        labels_mc = ["$b$", r"$\sigma(b)$", "$m$", "$c$", r"$\sigma(m)$","$Q$"]
        std = np.std(diff)
        start_params = np.array([lognuguess, 0.02, f[0], f[1], std, 0.5])
        bounds = [(lognuguess-.05, lognuguess+.05,), (0.01,0.05),\
                    (f[0]-0.02,f[0]+0.02), (f[1]-0.3,f[1]+0.3), \
                    (std*0.5,std*1.5), (0,1)]#(l[0]*0.5, l[0]*1.5),
                    # (0, 1)]

        Model = cModel(x, y, l[1], l[0])
        lnprior = cPrior.Prior(bounds)
        Like = cLikelihood.Likelihood(lnprior,Model)

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

        fig, ax = plt.subplots(2, sharex=True)
        pos = ax[0].scatter(x, y, s=5, c=fg_pp, cmap='Blues_r', vmin=0, vmax=1, zorder=1000,label='US'+str(US))
        fig.colorbar(pos,ax=ax[0],label='Foreground Posterior Probability')
        ax[0].plot(x,res[2]*x+res[3],c='r',zorder=1001,label='MCMC BG Fit')
        ax[0].set_title('Synthetic Pop. for undershoot efficiencey of '+US)
        ax[0].set_ylabel(r"$log_{10}$($T_{eff}$ (K))")
        ax[0].axvline(res[0],linestyle='--',c='r',label=r"RGBB location")
        ax[0].legend(loc='best',fancybox=True)

        fg_m = np.exp(Like.lnlike_fg(res))
        bg_m = np.exp(Like.lnlike_bg(res))
        # weights = np.ones_like(x)/float(len(x))
        hy, _, _ = ax[1].hist(x, bins=bins, color ='k', histtype='step')
        ax2 = ax[1].twinx()
        ax2.scatter(x,fg_m, c='cornflowerblue',alpha=.5,label='FG',s=5)
        ax2.scatter(x,bg_m, c='orange',alpha=.5,label='BG',s=5)
        ax[1].axvline(res[0],linestyle='--',c='r',label=r"$RGBB location")
        ax[1].set_title(r"Histogram in $log_{10}$($\nu_{max}$)")
        ax[1].set_xlabel(r"$log_{10}$($\nu_{max}$ ($\mu$Hz))")
        ax[1].set_ylabel('Counts')
        ax2.set_ylabel('Normalised Probability')
        ax2.set_ylim(0.,0.010)
        ax[1].legend(loc='best',fancybox=True)
        fig.tight_layout()
        fig.savefig('Output/Saniya_RGBB/results_'+US+'.png')
        plt.show()
        plt.close()
        # ax[1].scatter(x,fg_m/fg_m.max()*hy.max(),c='cornflowerblue',alpha=.5,label='FG',s=5)
        # ax[1].scatter(x,bg_m/bg_m.max()*hy.max(),c='orange',alpha=.5,label='BG',s=5)

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


        '''Plotting the straight line fit probabilities'''
        fn = f[1] + x*f[0]
        diff = y - fn
        stdd = np.std(diff)
        bg = np.exp(Like.lnlike_bg(res))
        bg = np.exp(Model.bg(res))
        scope = np.arange(1., 2.75, 0.25)
        lo = 0.75
        i = 0

        fig, ax = plt.subplots(2,sharex=True)
        for hi in scope:
            data = diff[x > lo]
            data = data[x < hi]
            dbg = bg[x > lo]
            dbg = dbg[x < hi]
            stdd = np.std(data)

            n, _,_ = ax[0].hist(data,bins=int(np.sqrt(len(data))),histtype='step',label=str(lo)+'-'+str(hi))
            ax[1].scatter(data, dbg, s=5, label=str(lo)+'-'+str(hi))

            lo = hi
        ax[0].legend(loc='best',fancybox=True)
        ax[1].legend(loc='best',fancybox=True)
        ax[1].set_xlabel(r"$T_{eff}$ - Straight Line Fit")
        ax[0].set_ylabel('Counts per bin')
        ax[1].set_ylabel(r"Background Probability")
        fig.tight_layout()
        fig.savefig('Output/Saniya_RGBB/results_fg_'+US+'.png')
        plt.close('all')

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
