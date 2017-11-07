# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Oliver J. Hall

import numpy as np
import pandas as pd
from pyqt_fit import kde
import glob
import sys
import corner as corner
from tqdm import tqdm
# import seaborn as sns
import emcee
import scipy.interpolate as interpolate
import scipy.integrate as integrate

import matplotlib
# import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'
\
from statsmodels.robust.scale import mad as mad_calc
from sklearn import linear_model
import scipy.stats as stats
import scipy.misc as misc

def get_values(frame):
    files = glob.glob('../data/Ben_Fun/*3*')
    dfC3 = pd.read_csv(files[0], sep=',')
    dfT3 = pd.read_csv(files[1], sep=',')
    files = glob.glob('../data/Ben_Fun/*6*')
    dfC6 = pd.read_csv(files[0], sep=',')
    dfT6 = pd.read_csv(files[1],sep=',')

    dfC = pd.concat([dfC3,dfC6])
    dfT = pd.concat([dfT3,dfT6])
    dfC = dfC6
    dfT = dfT6


    dfC.rename(columns={'Ks' : 'M_ks', 'J' : 'M_j', 'H' : 'M_h'} , inplace=True)
    dfT.rename(columns={'Kmag' : 'Ks', 'Jmag' : 'J', 'Hmag' : 'H'}, inplace=True)

    #Cardelli+1989
    Jcorr = 0.282
    Hcorr = 0.190
    Kcorr = 0.114

    #Finding K2 Apparent in Ks, TRILEGAL absolute in Ks
    dfC['Aks'] = Kcorr*dfC.Av
    dfC['Ks'] = dfC.M_ks + dfC['mu0'] + dfC.Aks
    dfT['Aks'] = Kcorr*dfT.Av
    dfT['M_ks'] = dfT.Ks - dfT['mu0'] - dfT.Aks

    #Finding K2 Apparent in J, TRILEGAL absolute in J
    dfC['Aj'] = Jcorr*dfC.Av
    dfC['J'] = dfC.M_j + dfC['mu0'] + dfC.Aj
    dfT['Aj'] = Jcorr*dfT.Av
    dfT['M_j'] = dfT.J - dfT['mu0'] - dfT.Aj

    #Finding K2 Apparent in H, TRILEGAL absolute in H
    dfC['Ah'] = Hcorr*dfC.Av
    dfC['H'] = dfC.M_h + dfC['mu0'] + dfC.Ah
    dfT['Ah'] = Hcorr*dfT.Av
    dfT['M_h'] = dfT.H - dfT['mu0'] - dfT.Ah

    #Calculing HR Diagram values
    dfC['L'] = 4*np.pi*(dfC.rad*695700e3)**2*5.67e-8*dfC.Teff**4
    dfC['logL'] = np.log10(dfC['L']/3.828e26)
    dfC['logT'] = np.log10(dfC.Teff)

    dfC = get_errors(dfC,type='true')
    dfT = get_errors(dfT,type='gaia')
    if frame=='K2':
        return dfC.M_ks.values, dfC.Ks.values, dfC.sig_M.values, dfC, dfT
    if frame=='TRILEGAL':
        return dfT.M_ks.values, dfT.Ks.values, dfT.sig_M.values, dfC, dfT


def get_errors(df, type='true'):
    if type=='gaia':
        df['pi_err'] = np.abs(10.e-3 + np.random.normal(0,0.5,len(df)) * 2.e-3) #mas

        df['pi'] = 1000/df['dist']                 #Getting all parallax (mas)
        df['sig_d'] = (1000*df['pi_err']/df['pi']**2)  #Propagating the Gaia error

        df['sig_mu'] = np.sqrt( (5*np.log10(np.e)/df['dist'])**2 * df['sig_d']**2 )
        df['sig_M'] = df['sig_mu']  #Until we incorporate errors on Aks and Ks

    if type=='true':
        diffU = df['Ks_68U'] - df.M_ks
        diffL = df.M_ks - df['Ks_68L']
        df['sig_M'] = np.sqrt(diffL**2 + diffU**2)


    return df

class Prior:
    def __init__(self, _bounds):
        self.bounds = _bounds

    def lnprior(self, p):
        if not all(b[0] < v < b[1] for v, b in zip(p, self.bounds)):
            return -np.inf
        return 0

    def __call__(self, p):
        prior = self.lnprior(p)
        return prior

# The "foreground" linear likelihood:
class Likelihood:
    def __init__(self,_x,_y,_xerr, _lnprior):
        self.x = _x
        self.y = _y
        self.xerr = _xerr
        self.lnprior = _lnprior

    def lnlike_fg(self, p):
        b, sigrc, _, o, sigo = p
        sig = np.sqrt(sigrc**2 + self.xerr**2)

        return -0.5 * ((self.x - b) / sig)**2 - np.log(sig)

# The "background" outlier likelihood:
    def lnlike_bg(self, p):
        _, _, Q, o, sigo = p
        sig = np.sqrt(sigo**2 + self.xerr**2)
        val = np.abs(self.x).max() + np.abs(self.x).min()

        xn = self.x + val
        on = np.abs(o)

        return -np.log(xn) -np.log(sig) - 0.5 * (np.log(xn) - on)**2/sig**2

# Full probabilistic model.
    def lnprob(self, p):
        b, sigrc, Q, o, sigo = p

        # First check the prior.
        lp = self.lnprior(p)
        if not np.isfinite(lp):
            return -np.inf, None

        # Compute the vector of foreground likelihoods and include the q prior.
        ll_fg = self.lnlike_fg(p)
        arg1 = ll_fg + np.log(Q)

        # Compute the vector of background likelihoods and include the q prior.
        ll_bg = self.lnlike_bg(p)
        arg2 = ll_bg + np.log(1.0 - Q)

        # Combine these using log-add-exp for numerical stability.
        ll = np.nansum(np.logaddexp(arg1, arg2))

        # We're using emcee's "blobs" feature in order to keep track of the
        # foreground and background likelihoods for reasons that will become
        # clear soon.
        return lp + ll

    def __call__(self, p):
        logL = self.lnprob(p)
        return logL

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
    output.to_csv('Output/Ben_K2/2dmix_results.csv')
    return results, stddevs

if __name__ == '__main__':
    plt.close('all')
####---SETTING UP DATA
    x, y, xerr, dfC, df= get_values('TRILEGAL')

####---PLOTTING INITIAL DATA
    fig5, ax5 = plt.subplots(2,2,sharex=True)
    ax5[0,0].scatter(dfC.M_ks,dfC.Ks,s=3, zorder=1000)
    ax5[0,0].errorbar(dfC.M_ks,dfC.Ks, xerr=dfC.sig_M ,fmt=",k",alpha=.1, ms=0, capsize=0, lw=1, zorder=999)
    ax5[0,0].set_title('K2 C3 and C6 data')
    ax5[0,0].set_xlabel("Aboluste Magnitude (Ks)")
    ax5[0,0].set_ylabel('Apparent Magnitude (Ks)')

    ax5[1,0].hist(dfC.M_ks, bins=int(np.sqrt(len(dfC.M_ks))), color='k', histtype='step', normed=1)
    ax5[1,0].set_title('Histogram in Absolute magnitude')

    ax5[0,1].scatter(df[df.label==3].M_ks,df[df.label==3].Ks,s=3,c='g',label='RGB',zorder=1000)
    ax5[0,1].scatter(df[df.label==4].M_ks,df[df.label==4].Ks,s=3,c='y',label='CHeB',zorder=1000)
    ax5[0,1].errorbar(df.M_ks,df.Ks, xerr=df.sig_M ,fmt=",k",alpha=.1, ms=0, capsize=0, lw=1, zorder=999)
    ax5[0,1].set_title('TRILEGAL sim of K2 C3 and C6')
    ax5[0,1].legend(loc='best',fancybox=True)

    ax5[1,1].hist(df.M_ks, bins=int(np.sqrt(len(df.M_ks))), color='k', histtype='step', normed=1)
    ax5[1,1].set_title('Histogram in Absolute magnitude')
    fig5.tight_layout()
    fig5.savefig('Output/Ben_K2/investigate_k2.png')
    plt.show(fig5)

####---SETTING UP AND RUNNING MCMC
####-----TRILEGAL RUN
    labels_mc = ["$b$", r"$\sigma(RC)$", "$Q$", "$o$", r"$\sigma(o)$"]
    bounds = [(-2.0,-1.4), (0.01,0.2), (0, 1), (0.0,2.0), (0.1, 2.)]
    start_params = np.array([-1.6, 0.1, 0.5, 0.1, 1.0])

    lnprior = Prior(bounds)
    Like = Likelihood(x, y, xerr, lnprior)

    # Initialize the walkers at a reasonable location.
    ntemps, ndims, nwalkers = 2, len(bounds), 32

    p0 = np.zeros([ntemps,nwalkers,ndims])
    for i in range(ntemps):
        for j in range(nwalkers):
            p0[i,j,:] = start_params * (1.0 + np.random.randn(ndims) * 0.0001)

    # Set up the sampler.
    sampler = emcee.PTSampler(ntemps, nwalkers, ndims, Like, lnprior,threads=2)

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
####-----TRILEGAL RUN
    corner.corner(chain, bins=35, labels=labels_mc)
    plt.savefig('Output/Ben_K2/TRILEGAL_corner.png')
    plt.close()

    print('Calculating posteriors...')
    norm = 0.0
    fg_pp = np.zeros(len(x))
    bg_pp = np.zeros(len(x))
    lotemp = sampler.chain[0,:,:,:]

    print('About to do posteriors...')
    for i in tqdm(range(lotemp.shape[0])):
        for j in range(lotemp.shape[1]):
            ll_fg = Like.lnlike_fg(lotemp[i,j])
            ll_bg = Like.lnlike_bg(lotemp[i,j])
            fg_pp += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
            bg_pp += np.exp(ll_bg - np.logaddexp(ll_fg, ll_bg))
            norm += 1
    fg_pp /= norm
    bg_pp /= norm

    lnK = np.log(fg_pp) - np.log(bg_pp)
    mask = lnK > 1

####---PLOTTING RESULTS
####-----TRILEGAL RUN
    print('Plotting results...')
    npa = chain.shape[1]
    res = np.zeros(npa)
    std = np.zeros(npa)
    for idx in np.arange(npa):
        res[idx] = np.median(chain[:,idx])
        std[idx] = np.std(chain[:,idx])

    b, sigrc, _, o, sigo = res
    sigf = np.sqrt(sigrc**2 + xerr**2)
    sigb = np.sqrt(sigo**2 + xerr**2)
    val = np.abs(x).max() + np.abs(x).min()
    xn = x + val
    on = np.abs(o)
    fg_m =  np.exp(-0.5 * ((x - b) / sigf)**2 - np.log(sigf))
    bg_m = np.exp(-np.log(xn) -np.log(sigb) - 0.5 * (np.log(xn) - on)**2/sigb**2)

    rc = b  #RC luminosity
    err = std[0]   #stddev on RC luminosity
    rcy = np.linspace(6,15,10)  #Y-axis for RC plot

    rcx = np.linspace(rc-err,rc+err,10) #Setting up shading bounds
    rcy1 = np.ones(rcx.shape) * y.max()
    rcy2 = np.ones(rcx.shape) * 6

    # Plot mixture model results
    plt.close('all')
    fig, ax = plt.subplots(2)
    ax[0].errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    s = ax[0].scatter(x, y, s=5, c=fg_pp, cmap="Blues_r", vmin=0, vmax=1, zorder=1000)
    ax[0].fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001,label='RC Confidence Interval')
    ax[0].set_title(r"Inlier Posterior Probabilities for TRILEGAL simulated data")
    ax[0].set_xlabel(r"$M_{Ks}$")
    ax[0].set_ylabel(r"$m_{Ks}$")
    fig.colorbar(s,label='Inlier Posterior Probability', ax = ax[0])
    ax[0].grid()
    ax[0].set_axisbelow(True)

    weights = np.ones_like(x)/float(len(x))
    hy, _, _ = ax[1].hist(x, weights=weights, bins=int(np.sqrt(len(x))), color='k', histtype='step')
    ax[1].fill_between(rcx,0.,hy.max(),color='red',alpha=0.8,zorder=1001,label='RC Confidence Interval')
    ax[1].scatter(x,fg_m/fg_m.max()*hy.max(),c='cornflowerblue',alpha=.5,label='FG',s=5)
    ax[1].scatter(x,bg_m/fg_m.max()*hy.max(),c='orange',alpha=.5,label='BG',s=5)
    ax[1].legend(loc='best',fancybox='True')
    ax[1].set_title('Histogram in Absolute magnitude')
    fig.tight_layout()
    plt.savefig('Output/Ben_K2/TRILEGAL_result-comp.png')
    plt.close('all')

    '''Getting KDEs'''
    kdes = []
    fig, ax = plt.subplots(chain.shape[1],chain.shape[1])
    for n in range(chain.shape[1]):
        est_large = kde.KDE1D(chain[:,n])
        xs, ys = est_large.grid()
        kdes.append(np.array([xs,ys]))
        a = ax[n,n].hist(chain[:,n],bins=int(np.sqrt(len(chain[:,n]))),histtype='step',color='k', normed=True)
        ax[n,n].plot(xs,ys,c='cornflowerblue')
        ax[n,n].set_title(labels_mc[n])
    fig.tight_layout()
    fig.savefig('Output/Ben_K2/KDE_fits.png')
    plt.close(fig)


####---SETTING UP AND RUNNING MCMC
####-----K2 RUN
    x, y, xerr, df, dfT= get_values('K2')

    start_params = res
    start_params[2] = 0.5
    bounds = [(res[0]-res[1],res[0]+res[1]), (res[1]-5*std[1],res[1]+5*std[1]),\
                (0, 1), (0.0, 2.0), (0.1, 2.)]

    # bounds = [(kdes[0][0,0],kdes[0][0,-1]),\
    #           (kdes[1][0,0],kdes[1][0,-1]),\
    #           (0.,1.),\
    #           (kdes[3][0,0],kdes[3][0,-1]),\
    #           (kdes[4][0,0],kdes[4][0,-1])]
    # if bounds[2][0] < 0.0:  #Q check
    #     bounds[2] = (0.0,bounds[2][1])

    lnprior = Prior(bounds)
    Like = Likelihood(x, y, xerr, lnprior)

    # Initialize the walkers at a reasonable location.
    ntemps, ndims, nwalkers = 2, len(bounds), 32

    p0 = np.zeros([ntemps,nwalkers,ndims])
    for i in range(ntemps):
        for j in range(ndims):
            xs, ys = kdes[j]
            cdf = integrate.cumtrapz(ys, xs,initial=0)
            inv_cdf = interpolate.interp1d(cdf, xs)
            p0[i,:,j] = inv_cdf(np.random.rand(nwalkers))

    # Set up the sampler.
    sampler = emcee.PTSampler(ntemps, nwalkers, ndims, Like, lnprior,threads=2)

    # Run a burn-in chain and save the final location.
    print('Burning in emcee...')
    for p1, lnpp, lnlp in tqdm(sampler.sample(p0, iterations=500)):
        pass

    # Run the production chain.
    print('Running emcee...')
    sampler.reset()
    for pp, lnpp, lnlp in tqdm(sampler.sample(p1, iterations=500)):
        pass
    chain = sampler.chain[0,:,:,:].reshape((-1, ndims))

####---CONSOLIDATING RESULTS
####-----K2 RUN
    corner.corner(chain, bins=35, labels=labels_mc)
    plt.savefig('Output/Ben_K2/K2_corner.png')
    plt.close()

    print('Calculating posteriors...')
    norm = 0.0
    fg_pp = np.zeros(len(x))
    bg_pp = np.zeros(len(x))
    lotemp = sampler.chain[0,:,:,:]

    print('About to do posteriors...')
    for i in tqdm(range(lotemp.shape[0])):
        for j in range(lotemp.shape[1]):
            ll_fg = Like.lnlike_fg(lotemp[i,j])
            ll_bg = Like.lnlike_bg(lotemp[i,j])
            fg_pp += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
            bg_pp += np.exp(ll_bg - np.logaddexp(ll_fg, ll_bg))
            norm += 1
    fg_pp /= norm
    bg_pp /= norm

    lnK = np.log(fg_pp) - np.log(bg_pp)
    k2mask = lnK > 1

####---PLOTTING RESULTS
####-----K2 RUN
    print('Plotting results...')
    npa = chain.shape[1]
    res = np.zeros(npa)
    std = np.zeros(npa)
    for idx in np.arange(npa):
        res[idx] = np.median(chain[:,idx])
        std[idx] = np.std(chain[:,idx])

    b, sigrc, _, o, sigo = res
    sigf = np.sqrt(sigrc**2 + xerr**2)
    sigb = np.sqrt(sigo**2 + xerr**2)
    val = np.abs(x).max() + np.abs(x).min()
    xn = x + val
    on = np.abs(o)
    fg_m =  np.exp(-0.5 * ((x - b) / sigf)**2 - np.log(sigf))
    bg_m = np.exp(-np.log(xn) -np.log(sigb) - 0.5 * (np.log(xn) - on)**2/sigb**2)

    k2rc = b  #RC luminosity
    k2err = std[0]   #stddev on RC luminosity
    k2rcy = np.linspace(6,15,10)  #Y-axis for RC plot

    k2rcx = np.linspace(k2rc-k2err,k2rc+k2err,10) #Setting up shading bounds
    k2rcy1 = np.ones(k2rcx.shape) * y.max()
    k2rcy2 = np.ones(k2rcx.shape) * 6

    # Plot mixture model results
    plt.close('all')
    fig, ax = plt.subplots(2)
    ax[0].errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    s = ax[0].scatter(x, y, s=5, c=fg_pp, cmap="Blues_r", vmin=0, vmax=1, zorder=1000)
    ax[0].fill_between(k2rcx,k2rcy1,k2rcy2,color='red',alpha=0.8,zorder=1001,label='RC Confidence Interval')
    ax[0].set_title(r"Inlier Posterior Probabilities for TRILEGAL simulated data")
    ax[0].set_xlabel(r"$M_{Ks}$")
    ax[0].set_ylabel(r"$m_{Ks}$")
    fig.colorbar(s,label='Inlier Posterior Probability', ax = ax[0])
    ax[0].grid()
    ax[0].set_axisbelow(True)

    weights = np.ones_like(x)/float(len(x))
    hy, _, _ = ax[1].hist(x, weights=weights, bins=int(np.sqrt(len(x))), color='k', histtype='step')
    ax[1].fill_between(k2rcx,0.,hy.max(),color='red',alpha=0.8,zorder=1001,label='RC Confidence Interval')
    ax[1].scatter(x,fg_m/fg_m.max()*hy.max(),c='cornflowerblue',alpha=.5,label='FG',s=5)
    ax[1].scatter(x,bg_m/fg_m.max()*hy.max(),c='orange',alpha=.5,label='BG',s=5)
    ax[1].legend(loc='best',fancybox='True')
    ax[1].set_title('Histogram in Absolute magnitude')
    fig.tight_layout()
    plt.savefig('Output/Ben_K2/K2-results-comp.png')
    plt.close('all')



####---SAVING STUFF INTO A PANDAS LIBRARY
    print('Saving output...')
    results, stddevs = save_library(df,chain,labels_mc)

    df["est_type"] = ""
    df.est_type[k2mask] = "RC"
    df.est_type[~k2mask] = "RGB"
    df.to_csv('Output/Ben_K2/K2C3_C6_RClabels.csv')
    dfTm = dfT[~mask]

####---PLOTTING OUTLIERS
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    ax[0].errorbar(x[~k2mask], y[~k2mask], xerr=xerr[~k2mask], fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    ax[0].scatter(x[~k2mask], y[~k2mask], s=5, c='cornflowerblue', zorder=1000)
    ax[0].fill_between(k2rcx,k2rcy1,k2rcy2,color='red',alpha=0.8,zorder=1001,label='RC Confidence Interval')
    ax[0].set_title(r'K2 C3 and C6 data ($~$no RC)')
    ax[0].set_xlabel("Absolute Magnitude (Ks)")
    ax[0].set_ylabel('Apparent Magnitude (Ks)')
    ax[0].legend(loc='best',fancybox=True)

    ax[1].scatter(dfTm[dfTm.label==3].M_ks,dfTm[dfTm.label==3].Ks,s=3,c='g',label='RGB')
    ax[1].scatter(dfTm[dfTm.label==4].M_ks,dfTm[dfTm.label==4].Ks,s=3,c='y',label='CHeB')
    ax[1].fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001,label='RC Confidence Interval')
    ax[1].errorbar(dfTm.M_ks, dfTm.Ks, fmt=',k', ms=0, capsize=0, lw=1, zorder=999)
    ax[1].set_title(r'TRILEGAL sim of K2 C3 and C6 ($~$no RC)')
    ax[1].legend(loc='best',fancybox=True)

    fig.tight_layout()
    fig.savefig('Output/Ben_K2/RC_removed.png')
    plt.show(fig)

    '''RC STATS FOR TRILEGAL'''
    rc_total = len(dfT[dfT.label==4])
    rc_captured = rc_total - len(dfTm[dfTm.label==4])
    total_mask = len(dfT[mask])
    rc_mask = len(dfT[mask][dfT[mask].label==4])
    rgb_mask = len(dfT[mask][dfT[mask].label==3])

    print('\nPercentage of RC stars in our mask: '+str(100.*rc_captured/rc_total) +'%')
    print('\nPercentage of mask contaminated with RGB: '+str(100.*rgb_mask/total_mask)+'%')
    print('\nWe thus have identified '+str(total_mask)+' stars with a '+str(100.*rc_mask/total_mask)+\
            '% confidence to belong to the Red Clump')
    print('\nThis applies to the TRILEGAL data, the fit to K2 data is likely less confident.')

    #Calculing HR Diagram values
    df['L'] = 4*np.pi*(df.rad*695700e3)**2*5.67e-8*df.Teff**4
    df['logL'] = np.log10(df['L']/3.828e26)
    df['logT'] = np.log10(df.Teff)

    '''HR Comparison'''
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_axes([0.1,0.2,0.35,0.6])
    ax2 = fig.add_axes([0.55,0.2,0.35,0.6],sharex=ax1,sharey=ax1)

    ax1.scatter(df[df.est_type=='RGB'].logT,df[df.est_type=='RGB'].logL,s=3,label='Sample with cut',zorder=1000)
    ax1.scatter(df.logT,df.logL,s=3,c='k',alpha=.1,label='Original',zorder=999)
    ax1.legend(loc='best',fancybox=True)
    ax1.set_title('K2 Campaign 3 data')
    ax1.invert_xaxis()
    ax1.set_xlabel(r"$log_{10}(T_{eff})$")
    ax1.set_ylabel(r'$log_{10}(L)$')

    ax2.scatter(dfTm[dfTm.label==3].logTe,dfTm[dfTm.label==3].logL,s=3,c='g',label='RGB',zorder=1000)
    ax2.scatter(dfTm[dfTm.label==4].logTe,dfTm[dfTm.label==4].logL,s=3,c='y',label='CHeB',zorder=1000)
    ax2.scatter(dfT.logTe,dfT.logL,s=1,c='k',alpha=.1,label='Original',zorder=999)
    ax2.set_title('TRILEGAL sim of K2 C3')
    ax2.legend(loc='best',fancybox=True)
    fig.savefig('Output/Ben_K2/HR_comparison.png')
    plt.show()

# #__________________CALCULATING OUTLIER FRACTION____________________________
#     '''Red clump label: 4 | RGB label: 3'''
#     rctotal = tortoise(df[mask])
#     rctotal_mod = len(df[labels==4])
#     rcm = len(df[mask][labels==4])
#     rgbm = len(df[mask][labels==3])
#     starm = len(np.where(mask==True)[0])
#
#     print('\nTotal number of RC stars in the reduced sample: '+str(rctotal_mod)+', '+str(rctotal_mod*100/rctotal)+ '% of the total sample.')
#     print('Number of stars in mask: '+str(starm))
#     print('Percentage of which are RC: '+str(rcm*100/starm)+'%')
#     print('The remaining stars are all RGB stars.')
#     print('\nFraction of all RC stars identified: ' +str(rcm*100/rctotal_mod)+'%')


####---CODE GRAVEYARD

    # #Plot labeled results
    # fig, ax = plt.subplots()
    # ax.scatter(x[~mask][labels[~mask]==3], y[~mask][labels[~mask]==3], c=c[3], s=1,zorder=1000,label=label[3])
    # ax.scatter(x[~mask][labels[~mask]==4], y[~mask][labels[~mask]==4], c=c[4], s=1,zorder=1000,label=label[4])
    # ax.errorbar(x[~mask], y[~mask], xerr=xerr[~mask], fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    # ax.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001,label='RC Confidence Interval')
    #
    # ax.set_xlabel(r"$M_{Ks}$")
    # ax.set_ylabel(r"$m_{Ks}$")
    # ax.set_title(r"TRILEGAL simulated data with classified RC stars removed")
    # ax.legend(loc='best',fancybox=True)
    # ax.grid()
    # ax.set_axisbelow(True)
    #
    # plt.show()
    # plt.close('all')
