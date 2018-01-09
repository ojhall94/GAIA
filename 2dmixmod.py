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
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

from statsmodels.robust.scale import mad as mad_calc
from sklearn import linear_model
import scipy.stats as stats
import scipy.misc as misc

import cMCMC
import cLikelihood
import cPrior

def get_values():
    sfile = glob.glob('../Cuts_Data/cuts_MH_JKs_logg.txt')[0]
    try:
        df = pd.read_csv(sfile, sep='\s+')
    except IOError:
        print('Cuts file doesnt exist, run the slider first.')

    '''Errors not included in current run'''
    # df = get_errors(df, DR=2)
    df = df[0:10000]

    return df.M_ks.values, df.Ks.values, df.stage, df

def get_errors(df, DR=2):
    if DR == 2:
        df['pi_err'] = np.abs(0.2 + np.random.normal(0,0.5,len(df)) * 0.1) #mas

    if DR == 'fin':
        df['pi_err'] = np.abs(10.e-3 + np.random.normal(0,0.5,len(df)) * 2.e-3) #mas

    df['d'] = 10**(1+ (df['m-M0']/5))       #Getting all distances (pc)
    df['pi'] = 1000/df['d']                 #Getting all parallax (mas)
    df['sig_d'] = (1000*df['pi_err']/df['pi']**2)  #Propagating the Gaia error

    df['sig_mu'] = np.sqrt( (5*np.log10(np.e)/df['d'])**2 * df['sig_d']**2 )
    df['sig_M'] = df['sig_mu']  #Until we incorporate errors on Aks and Ks

    plt.errorbar(df['pi'],df['Ks'],xerr=df['pi_err'],alpha=.1,fmt=".k",c='grey',zorder=999)
    plt.scatter(df['pi'],df['Ks'],s=5,zorder=1000)
    plt.show()

    plt.errorbar(df['d'],df['Ks'],xerr=df['sig_d'],alpha=.1,fmt=".k",c='grey',zorder=999)
    plt.scatter(df['d'],df['Ks'],s=5,zorder=1000)
    plt.show()
    return df


def probability_plot(x, y, X, Y, d, bins):
    #Plotting residuals with histograms
    left, bottom, width, height = 0.1, 0.35, 0.60, 0.60
    fig = plt.figure(1, figsize=(8,8))
    sax = fig.add_axes([left, bottom, width, height])
    yax = fig.add_axes([left+width+0.02, bottom, 0.25, height])
    xax = fig.add_axes([left, 0.1, width, 0.22], sharex=sax)
    sax.xaxis.set_visible(False)
    yax.set_yticklabels([])
    yax.set_xticklabels([])
    xax.set_yticklabels([])
    xax2 = xax.twinx()
    xax2.set_yticklabels([])
    xax.grid()
    xax.set_axisbelow(True)
    yax.grid()
    yax.set_axisbelow(True)

    fig.suptitle('Probability functions to be applied to TRILEGAL data.')

    sax.hist2d(x, y,bins=bins, cmap='Blues_r', zorder=1000)
    sax.contour(X,Y,d, cmap='copper',alpha=.5,label='KDE',zorder=1001)

    yax.hist(y,bins=bins,color='r',histtype='step',orientation='horizontal', normed=True)
    yax.set_ylim(sax.get_ylim())
    yax.legend(loc='best')

    xax.hist(x,bins=bins,histtype='step',color='r',normed=True)

    xax.set_xlabel(r"$M_{Ks}$")
    sax.set_ylabel(r"$m_{Ks}$")

    return fig


class cModel:
    '''Models for this run.'''
    def __init__(self, _x, _y, _xerr):
        self.x = _x
        self.y = _y
        self.xerr = _xerr

    def fg(self, p):
        b, sigrc, _, _, _ = p
        sig = np.sqrt(sigrc**2 + self.xerr**2)
        return -0.5 * ((self.x - b) / sig)**2 - np.log(sig)

    def bg(self, p):
        _, _, o, sigo, _ = p
        sig = np.sqrt(sigo**2 + self.xerr**2)
        val = np.abs(self.x).max() + np.abs(self.x).min()

        xn = self.x + val
        on = np.abs(o)

        return -np.log(xn) -np.log(sig) - 0.5 * (np.log(xn) - on)**2/sig**2

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
    output.to_csv('../Output/2dmix_results.csv')
    df.to_csv('../Output/2dmix_selected_data.csv')

    return results, stddevs

def tortoise(dfm):
    sfile = glob.glob('../data/TRILEGAL_sim/*all*.txt')[0]
    df = pd.read_csv(sfile, sep='\s+')
    labels = df['stage'].values
    Zish = df['[M/H]'].values
    logT = df['logTe'].values
    logL = df['logL'].values

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots(3,3)
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    loc = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]

    for i in range(int(np.nanmax(labels))+1):
        ax.scatter(logT[labels==i],logL[labels==i],s=5,c=c[i],label=str(i))
        im = ax2[loc[i]].scatter(logT[labels==i],logL[labels==i],s=10,c=Zish[labels==i],\
                            cmap='viridis',vmin=-4,vmax=0.7)#vmin=0.0,vmax=4.)
        ax2[loc[i]].scatter(dfm.logTe[dfm.stage==i],dfm.logL[dfm.stage==i],s=7,c='r',marker=',')
        ax2[loc[i]].set_title(str(i))
        ax2[loc[i]].invert_xaxis()

    ax.scatter(dfm['logTe'],dfm['logL'],s=5,c='r',marker=',',label='Mask')
    ax.legend(loc='best')
    ax.invert_xaxis()
    ax.set_xlabel('log(T)')
    ax.set_ylabel('log(L)')
    fig2.tight_layout()

    fig2.subplots_adjust(right=0.8)
    cbar_ax = fig2.add_axes([0.85,0.15,0.05,0.7])
    fig2.colorbar(im,cax=cbar_ax,label='[M/H]')

    fig.tight_layout()
    fig.savefig('../Output/HR_mask.png')
    fig2.savefig('../Output/HRsplit_mask.png')
    plt.show()
    return len(df[labels==4])

if __name__ == '__main__':
    plt.close('all')
####---SETTING UP DATA
    x, y, labels, df = get_values()

####---PLOTTING INITIAL DATA
    fig, ax = plt.subplots()
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    label = ['Pre-Main Sequence', 'Main Sequence', 'Subgiant Branch', 'Red Giant Branch', 'Core Helium Burning',\
                'RR Lyrae variables', 'Cepheid Variables', 'Asymptotic Giant Branch','Supergiants']
    ax.scatter(x[labels==3], y[labels==3], c=c[3], s=1,zorder=1000,label=label[3])
    ax.scatter(x[labels==4], y[labels==4], c=c[4], s=1,zorder=1001,label=label[4])
    ax.set_xlabel(r"$M_{Ks}$")
    ax.set_ylabel(r"$m_{Ks}$")
    ax.set_title("Labeled TRILEGAL simulated data cut in [M/H], logg & J-Ks")
    ax.legend(loc='best',fancybox=True)
    ax.grid()
    ax.set_axisbelow(True)
    fig.savefig('Output/investigate_TRILEGAL.png')
    plt.show()
    plt.close('all')

####---BUILDING KDE AND OTHER PARAM
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

    bins = int(np.sqrt(len(x)))
    fig = probability_plot(x, y, X, Y, d, bins)
    plt.show()
    sys.exit()

####---SETTING UP MCMC
    labels_mc = ["$b$", r"$\sigma(RC)$", "$o$", r"$\sigma(o)$", "$Q$"]
    bounds = [(-1.7,-1.4), (0.01,0.2), (0.0,2.0), (0.1, 2.), (0, 1)]
    start_params = np.array([-1.6, 0.1, 0.1, 1.0, 0.5])


####---CHECKING MODELS BEFORE RUN
    #Getting other probability functions
    ModeLLs = cLLModels.LLModels(x, y, labels_mc)
    '''NEED TO CORRECTLY SET UP THE MODEL STRUCTURE FOR THIS'''

    fig = probability_plot(x, y, fy, X, Y, bins, '''REMAINDER''')
    fig.savefig('Output/visual_models.png')
    plt.show()
    plt.close('all')

####---RUNNING MCMC
    ModeLLs = cLLModels.LLModels(x, y, labels_mc)
    lnpiror = cPrior.Prior(bounds)
    Like = cLikelihood(lnprior, ModeLLs)

    ntemps, nwalkers = 4, 32

    Fit = cMCMC.MCMC(start_params, Like, lnprior, 'none', ntemps, 1000, nwalkers)
    chain = Fit.run()

####---CONSOLIDATING RESULTS
    corner.corner(chain, bins=35, labels=labels_mc)
    plt.savefig('../Output/corner.png')
    plt.show()
    plt.close()

    lnK, fg_pp = Fit.log_bayes()
    mask = lnK > 1
    Fit.dump()

####---PLOTTING RESULTS
    print('Plotting results...')
    rc = np.median(chain[:,0])  #RC luminosity
    err = np.std(chain[:,0])    #stddev on RC luminosity
    rcy = np.linspace(6,15,10)  #Y-axis for RC plot

    rcx = np.linspace(rc-err,rc+err,10) #Setting up shading bounds
    rcy1 = np.ones(rcx.shape) * y.max()
    rcy2 = np.ones(rcx.shape) * 6

    # Plot mixture model results
    fig, ax = plt.subplots()
    ax.errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    s = ax.scatter(x, y, marker="1", s=1, c=fg_pp, cmap="Blues_r", vmin=0, vmax=1, zorder=1000)
    ax.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001,label='RC Confidence Interval')
    ax.set_title(r"Inlier Posterior Probabilities for TRILEGAL simulated data")
    ax.set_xlabel(r"$M_{Ks}$")
    ax.set_ylabel(r"$m_{Ks}$")
    fig.colorbar(s,label='Inlier Posterior Probability')
    ax.legend(loc='best',fancybox='True')

    ax.grid()
    ax.set_axisbelow(True)
    plt.savefig('/home/oliver/Dropbox/Papers/Midterm/Images/C4_posteriors.png')
    # plt.savefig('../Output/posteriors.png')
    plt.show()
    plt.close('all')

    #Plot labeled results
    fig, ax = plt.subplots()
    ax.scatter(x[~mask][labels[~mask]==3], y[~mask][labels[~mask]==3], c=c[3], s=1,zorder=1000,label=label[3])
    ax.scatter(x[~mask][labels[~mask]==4], y[~mask][labels[~mask]==4], c=c[4], s=1,zorder=1000,label=label[4])
    ax.errorbar(x[~mask], y[~mask], xerr=xerr[~mask], fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    ax.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001,label='RC Confidence Interval')

    ax.set_xlabel(r"$M_{Ks}$")
    ax.set_ylabel(r"$m_{Ks}$")
    ax.set_title(r"TRILEGAL simulated data with classified RC stars removed")
    ax.legend(loc='best',fancybox=True)
    ax.grid()
    ax.set_axisbelow(True)

    plt.savefig('/home/oliver/Dropbox/Papers/Midterm/Images/C4_result.png')
    # plt.savefig('../Output/dataset.png')
    plt.show()
    plt.close('all')



####---SAVING STUFF INTO A PANDAS LIBRARY
    print('Saving output...')
    results, stddevs = save_library(df,chain,labels_mc)

    plt.scatter(x,np.exp(lnlike_fg(results.loc[0].values)))
    plt.savefig('../Output/fg_like.png')
    plt.close()
    plt.scatter(x, np.exp(lnlike_bg(results.loc[0].values)))
    plt.savefig('../Output/bg_like.png')
    plt.close()

#__________________GETTING CORRECTION____________________________
    print('Calculating corrections...')
    #Plot showing correction
    m_ks = df.Ks.values[mask]
    Aks = df.Aks.values[mask]
    pi_o = df['pi'].values[mask]
    pi_o_err = df['pi_err'].values[mask]

    #Calculating RC parallax from our fit value for magnitude
    Mrc = np.ones(y[mask].shape)*rc                 #Fit RC luminosity
    Mrcerr = np.ones(y[mask].shape)*err             #Fit RC error
    mu_rc = m_ks - Mrc - Aks                        #Calculating distance modulus

    d_rc = 10**(1+mu_rc/5)      #Calculating distance
    pi_rc = 1000/d_rc            #Calculating parallax

    d_err = np.sqrt( (2*np.log(10)*10**(mu_rc/5))**2 * Mrcerr**2 )
    pi_rc_err = 1000 * d_err / d_rc**2

    #Fit a straight line to the data
    fit = np.polyfit(pi_rc, pi_o, 1)
    fn = np.poly1d(fit)

    #Plotting results
    fig, ax = plt.subplots(2)
    ax[0].errorbar(pi_rc,pi_o, xerr=pi_rc_err, yerr=pi_o_err,fmt=",k",alpha=.1, ms=0, capsize=0, lw=1, zorder=999)
    im = ax[0].scatter(pi_rc,pi_o, marker="s", s=5, c=m_ks, cmap="viridis", zorder=1000)
    ax[0].plot(pi_rc,pi_rc,linestyle='--',c='r',alpha=.5, zorder=1001, label='One to One')
    ax[0].plot(pi_rc, fn(pi_rc),c='k',alpha=.9, zorder=1002, label='Straight line polyfit')
    ax[0].set_xlabel('RC parallax (mas)')
    ax[0].set_ylabel('TRILEGAL parallax (mas)')
    ax[0].set_title(r'Determined RC parallax compared to TRILEGAL parallax for classified RC stars')
    ax[0].legend(loc='best')

    yrr = np.sqrt( (pi_o_err / pi_rc)**2 + ( (pi_o/pi_rc**2)**2 * pi_rc_err**2))
    # yrr = np.sqrt(pi_rc_err**2 + pi_o_err**2)
    ax[1].errorbar(pi_rc,pi_o/pi_rc, xerr=pi_rc_err,yerr=yrr, fmt=",k",alpha=.2, ms=0, capsize=0, lw=1, zorder=999)
    ax[1].scatter(pi_rc,pi_o/pi_rc, s=1,c=m_ks,cmap='viridis',zorder=1000)
    ax[1].set_xlabel('RC parallax (mas)')
    ax[1].set_ylabel('TRILEGAL parallax/RC parallax')
    ax[1].set_title(r'Residuals to one-to-one for upper plot')
    ax[1].axhline(y=1.0,c='r',alpha=.5,linestyle='--')
    fig.tight_layout()

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
    norm = matplotlib.colors.Normalize(vmin=m_ks.min(),vmax=m_ks.max())
    col = matplotlib.colorbar.ColorbarBase(cbar_ax,cmap='viridis',norm=norm,orientation='vertical',label='TRILEGAL apaprent magnitude')

    ax[0].grid()
    ax[0].set_axisbelow(True)
    ax[1].grid()
    ax[1].set_axisbelow(True)

    plt.savefig('/home/oliver/Dropbox/Papers/Midterm/Images/C4_corrections.png')
    # plt.savefig('../Output/corrections.png')
    plt.show()

#__________________CALCULATING OUTLIER FRACTION____________________________
    '''Red clump label: 4 | RGB label: 3'''
    rctotal = tortoise(df[mask])
    rctotal_mod = len(df[labels==4])
    rcm = len(df[mask][labels==4])
    rgbm = len(df[mask][labels==3])
    starm = len(np.where(mask==True)[0])

    print('\nTotal number of RC stars in the reduced sample: '+str(rctotal_mod)+', '+str(rctotal_mod*100/rctotal)+ '% of the total sample.')
    print('Number of stars in mask: '+str(starm))
    print('Percentage of which are RC: '+str(rcm*100/starm)+'%')
    print('The remaining stars are all RGB stars.')
    print('\nFraction of all RC stars identified: ' +str(rcm*100/rctotal_mod)+'%')
