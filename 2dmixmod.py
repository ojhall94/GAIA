# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Oliver J. Hall

import numpy as np
import pandas as pd
import corner as corner
import emcee

import glob
import sys
import ClosePlots as cp
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

import cMCMC
import cPrior
import cLLModels

'''<Unused in current build.>'''
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
    output.to_csv('Output/2dmix_results.csv')
    df.to_csv('Output/2dmix_selected_data.csv')

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
'''</unused in current build.>'''

def get_values():
    sfile = glob.glob('../Cuts_Data/cuts_MH_JKs_logg.txt')[0]
    try:
        df = pd.read_csv(sfile, sep='\s+')
    except IOError:
        print('Cuts file doesnt exist, run the slider first.')

    '''Errors not included in current run'''
    # df = get_errors(df, DR=2)
    df = df[(df.M_ks > -2.0) & (df.M_ks < -1.0)]
    df = df[0:10000]

    return df.M_ks.values, df.M_j.values, df.stage, df

class cLikelihood:
    '''A likelihood function that pulls in log likehoods from the LLModels class
    '''
    def __init__(self,_lnprior, _Model):
        self.lnprior = _lnprior
        self.Model = _Model

    #Likelihood for the 'foreground'
    def lnlike_fg(self, p):
        return self.Model.lorentzian(p[0:2]) + self.Model.lorentzian(p[4:6],dim='y')
        # return self.Model.lorentzian(p[0:2]) + self.Model.lorentzian(p[4:6],dim='y')

    def lnlike_bg(self, p):
        # return self.Model.lorentzian(p[2:4]) + self.Model.lorentzian(p[6:8],dim='y')
        return self.Model.lorentzian(p[2:4]) + self.Model.lorentzian(p[6:8],dim='y')

    def lnprob(self, p):
        Q = p[-1]

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

        return lp + ll

    def __call__(self, p):
        logL = self.lnprob(p)
        return logL

def probability_plot(x, y, df, bins, Ksx0, Jx0, lor_Ks_fg, lor_Ks_bg, lor_J_fg, lor_J_bg):
    #Plotting residuals with histograms
    left, bottom, width, height = 0.1, 0.35, 0.4, 0.60
    fig = plt.figure(1, figsize=(10,8))
    lax = fig.add_axes([left, bottom, width, height])           #Left-hand plot
    rax = fig.add_axes([left+width+0.05, bottom, width, height]) #Right-hand plot
    xax = fig.add_axes([left, 0.1, width, 0.22], sharex=lax)    #x histogram
    yax = fig.add_axes([left+width+0.05, 0.1, width, 0.22], sharex=rax) #y histogram
    #Clearing necessary axes
    lax.xaxis.set_visible(False)
    rax.xaxis.set_visible(False)
    yax.set_yticklabels([])
    yax2 = yax.twinx()
    yax2.set_yticklabels([])
    xax.set_yticklabels([])
    xax2 = xax.twinx()
    xax2.set_yticklabels([])
    #Turning on grid
    xax.grid()
    xax.set_axisbelow(True)
    yax.grid()
    yax.set_axisbelow(True)

    fig.suptitle('Probability functions to be applied to TRILEGAL data.')
    #Plotting left hand 2d hist
    lax.hist2d(x, df.Ks,bins=bins, cmap='Blues_r', zorder=1000)
    lax.axvline(Ksx0,c='r',zorder=1001)

    #Plotting right hand 2d hist
    rax.hist2d(y, df.J, bins=bins, cmap='Blues_r',zorder=1000)
    rax.axvline(Jx0, c='r', zorder=1001)

    #Plotting histogram with models for J
    yax.hist(y,bins=bins,histtype='step',color='r',normed=True)
    yax2.scatter(y,lor_J_fg,s=5,c='cornflowerblue',alpha=.5,label='Foreground Lorentzian in J')
    yax2.scatter(y,lor_J_bg,s=5,c='orange',alpha=.5,label='Background Lorentzian in J')
    yax2.scatter(y,lor_J_bg+lor_J_fg,s=1,c='green',alpha=.2,label='Combined')
    yax2.set_ylim(0.)
    yax.axvline(Jx0,c='r',label=r"$Jx0$")
    h1, l1 = yax.get_legend_handles_labels()
    h2, l2 = yax2.get_legend_handles_labels()
    yax.legend(h1+h2, l1+l2)

    #Plotting histograms with models for Ks
    xax.hist(x,bins=bins,histtype='step',color='r',normed=True)
    xax2.scatter(x,lor_Ks_fg,s=5,c='cornflowerblue',alpha=.5,label='Foreground Lorentzian in Ks')
    xax2.scatter(x,lor_Ks_bg,s=5,c='orange',alpha=.5,label='Background Lorentzian in Ks')
    xax2.scatter(x,lor_Ks_fg+lor_Ks_bg,s=1,c='green',alpha=.2,label='Combined')
    xax2.set_ylim(0.)
    xax.axvline(Ksx0,c='r',label=r"$Ksx0$")
    h1, l1 = xax.get_legend_handles_labels()
    h2, l2 = xax2.get_legend_handles_labels()
    xax.legend(h1+h2, l1+l2)

    #Set labels
    xax.set_xlabel(r"$M_{Ks}$")
    yax.set_xlabel(r"$M_{J}$")
    lax.set_ylabel(r"$m_{Ks}$")
    rax.set_ylabel(r"$m_{J}$")
    return fig


if __name__ == '__main__':
    plt.close('all')
####---SETTING UP DATA
    x, y, labels, df = get_values()

####---PLOTTING INITIAL DATA
    fig, ax = plt.subplots()
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    label = ['Pre-Main Sequence', 'Main Sequence', 'Subgiant Branch', 'Red Giant Branch', 'Core Helium Burning',\
                'RR Lyrae variables', 'Cepheid Variables', 'Asymptotic Giant Branch','Supergiants']
    ax.scatter(x[labels==3], df.Ks[labels==3], c=c[3], s=1,zorder=1000,label=label[3])
    ax.scatter(x[labels==4], df.Ks[labels==4], c=c[4], s=1,zorder=1001,label=label[4])
    sel = (labels!=3)&(labels!=4)
    ax.scatter(x[sel], df.Ks[sel], c=c[0], s=1, zorder=1002, label='Other types')
    ax.set_xlabel(r"$M_{Ks}$")
    ax.set_ylabel(r"$m_{Ks}$")
    ax.set_title("Labeled TRILEGAL simulated data cut in [M/H], logg & J-Ks")
    ax.legend(loc='best',fancybox=True)
    ax.grid()
    ax.set_axisbelow(True)
    fig.savefig('Output/investigate_TRILEGAL.png')
    plt.close('all')

####---EXAMINING & ESTIMATING PARAMETERS
    bins = int(np.sqrt(len(x)))

    n, b = np.histogram(x,bins=bins)
    Ksx0guess = b[np.argmax(n)]

    n, b = np.histogram(y, bins=bins)
    Jx0guess = b[np.argmax(n)]

####---SETTING UP MCMC
    style = 'lors'
    if style == 'lors':
        labels_mc = ["$x0 (Ks)$", r"$\gamma0$(Ks)",\
                    "$x1$(Ks)", r"$\gamma1$(Ks)",\
                    "$x0 (J)$", r"$\gamma0$(J)",\
                    "$x1$(J)", r"$\mu$(J)",\
                    "$Q$"]
        start_params = np.array([Ksx0guess, 0.02,   #Foreground params in Ks
                                -1.56, 0.1,         #Background params in Ks
                                Jx0guess, 0.05,     #Foreground params in J
                                -0.93, 0.1,         #Background params in J
                                0.5])              #Mixture model param
        bounds = [(-1.7,-1.6), (0.01,0.8),
                    (-1.6,-1.5), (0.08, 0.2),
                    (-1.1,-0.9), (0.01,0.8),
                    (-0.96,-0.9), (0.08, 1.2),
                    (0, 1)]
    if style == 'notlors':
        labels_mc = ["$x0 (Ks)$", r"$\gamma$ (Ks)",\
                    r"$\mu$(Ks)", r"$\sigma$(Ks)",\
                    "$x0 (J)$", r"$\gamma$ (J)",\
                    r"$\mu$(Ks)", r"$\sigma$(Ks)",\
                    "$Q$"]
        start_params = np.array([Ksx0guess, 0.02,   #Foreground params in Ks
                                -1.58, 0.14,         #Background params in Ks
                                Jx0guess, 0.05,     #Foreground params in J
                                -0.93, 0.1,         #Background params in J
                                0.5])              #Mixture model param
        bounds = [(-1.7,-1.63), (0.01,0.8),
                    (-1.63,-1.5), (0.08, 0.2),
                    (-1.1,-1.0), (0.01,0.8),
                    (-0.96,-0.9), (0.08, 1.2),
                    (0, 1)]

####---CHECKING MODELS BEFORE RUN
    #Getting other probability functions
    ModeLLs = cLLModels.LLModels(x, y, labels_mc)
    lor_Ks_fg = np.exp(ModeLLs.lorentzian(start_params[0:2]))
    lor_Ks_bg = np.exp(ModeLLs.lorentzian(start_params[2:4]))
    # lor_Ks_bg = np.exp(ModeLLs.gaussian(start_params[2:4]))
    lor_J_fg = np.exp(ModeLLs.lorentzian(start_params[4:6], dim='y'))
    lor_J_bg = np.exp(ModeLLs.lorentzian(start_params[6:8], dim='y'))
    # lor_J_bg = np.exp(ModeLLs.gaussian(start_params[6:8], dim='y'))

    #Visualising the probability distributions
    fig = probability_plot(x, y, df, bins, Ksx0guess, Jx0guess, lor_Ks_fg, lor_Ks_bg, lor_J_fg, lor_J_bg)
    fig.savefig('Output/visual_models.png')
    plt.show()
    plt.close('all')

####---RUNNING MCMC
    lnprior = cPrior.Prior(bounds)
    Like = cLikelihood(lnprior, ModeLLs)
    if np.isinf(lnprior(start_params)):
        print('Starting guesses out of bounds.')
        sys.exit()

    ntemps, nwalkers = 4, 32

    Fit = cMCMC.MCMC(start_params, Like, lnprior, 'none', ntemps, 1000, nwalkers)
    chain = Fit.run()

####---CONSOLIDATING RESULTS
    corner.corner(chain, bins=35, labels=labels_mc)
    plt.savefig('Output/corner.png')
    plt.show()
    plt.close()

    print('Plotting model results...')
    npa = chain.shape[1]
    res = np.zeros(npa)
    std = np.zeros(npa)
    for idx in np.arange(npa):
        res[idx] = np.median(chain[:,idx])
        std[idx] = np.std(chain[:,idx])

    #Calling probability functions with results
    lor_Ks_fg = np.exp(ModeLLs.lorentzian(res[0:2]))
    # lor_Ks_bg = np.exp(ModeLLs.lorentzian(res[2:4]))
    lor_Ks_bg = np.exp(ModeLLs.gaussian(res[2:4]))
    lor_J_fg = np.exp(ModeLLs.lorentzian(res[4:6], dim='y'))
    # lor_J_bg = np.exp(ModeLLs.lorentzian(res[6:8], dim='y'))
    lor_J_bg = np.exp(ModeLLs.gaussian(res[6:8], dim='y'))

    fig = probability_plot(x, y, df, bins, res[0], res[4], lor_Ks_fg, lor_Ks_bg, lor_J_fg, lor_J_bg)
    fig.savefig('Output/visual_result.png')

    print('Calculating posteriors...')
    lnK, fg_pp = Fit.log_bayes()
    sys.exit()

    #Calculate recall vs precision for various Kass+Raftery94 cut off scales
    recall, precision = [], []
    for lim in np.linspace(lnK.min(),lnK.max(),1000):
        mask = lnK > lim
        cheb_correct = len(x[mask][labels[mask]==4])
        cheb_total = len(x[labels==4])
        identified_total = len(x[mask])
        recall.append(float(cheb_correct)/float(cheb_total))
        try:
            precision.append(float(cheb_correct)/float(identified_total))
        except:
            precision.append(0.)
    f, a = plt.subplots()
    col = a.scatter(recall, precision, c=np.linspace(lnK.min(),lnK.max(),1000))
    a.axhline(0.9,c='r',label='0.9 precision')
    a.set_xlabel('Recall')
    a.set_ylabel('Precision')
    a.grid()
    a.set_axisbelow(True)
    f.colorbar(col, label='lnK cut-off point')
    f.savefig('Output/precisionvsrecall.png')
    plt.show()

    mask = lnK > 3

####---PLOTTING RESULTS

    # Plot mixture model results
    fig, ax = plt.subplots()
    ax.grid()
    ax.set_axisbelow(True)

    col = ax.scatter(x, df.Ks, c=fg_pp,s=5,cmap='viridis')
    ax.axvline(res[0],c='r',label=r"$\mu$")
    ax.set_title(r"Inlier Posterior Probabilities for TRILEGAL simulated data")
    ax.set_xlabel(r"$M_{Ks}$")
    ax.set_ylabel(r"$m_{Ks}$")
    fig.colorbar(col,label='Inlier Posterior Probability')
    ax.legend(loc='best',fancybox='True')
    fig.savefig('Output/posteriors.png')


    #Plotting identified results
    ffig, aax = plt.subplots()
    aax.scatter(x[~mask], df.Ks[~mask], c=c[3], s=1,zorder=1000,label='Outliers (RGB)')
    aax.scatter(x[mask], df.Ks[mask], c=c[4], s=1,zorder=1000,label='Inliers (RC)')
    aax.legend(loc='best',fancybox=True)

    cheb_correct = len(x[mask][labels[mask]==4])
    cheb_total = len(x[labels==4])
    identified_total = len(x[mask])
    recall = float(cheb_correct)/float(cheb_total)
    precision = float(cheb_correct)/float(identified_total)

    aax.set_xlabel(r"$M_{Ks}$")
    aax.set_ylabel(r"$m_{Ks}$")
    aax.set_title('Recall: '+str.format('{0:.2f}',recall) + '| Precision: '+str.format('{0:.2f}',precision))
    aax.legend(loc='best',fancybox=True)
    aax.grid()
    aax.set_axisbelow(True)

    ffig.savefig('Output/dataset.png')
    cp.show()
    plt.close('all')

    sys.exit()
    Fit.dump()

    '''Everything below here is redundant in current build.'''



####---SAVING STUFF INTO A PANDAS LIBRARY
    print('Saving output...')
    results, stddevs = save_library(df,chain,labels_mc)

    sys.exit()
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
