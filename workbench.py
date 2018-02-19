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
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

from statsmodels.robust.scale import mad as mad_calc
from sklearn import linear_model
import scipy.stats as stats
import scipy.misc as misc

import ClosePlots as cp

def get_values():
    # sfile = glob.glob('../data/TRILEGAL_sim/*.all.*.txt')[0]
    sfile = glob.glob('../data/TRILEGAL_sim/*all*.txt')[0]
    # sfile = glob.glob('../data/Ben_Fun/TRI3*')[0]
    df = pd.read_csv(sfile, sep='\s+')

    '''This function corrects for extinction and sets the RC search range'''
    df['Aks'] = 0.114*df.Av #Cardelli+1989>-
    df['M_ks'] = df.Ks - df['m-M0'] - df.Aks
    #
    corrections = pd.DataFrame(columns=['M_ks<','M_ks>','Ks<','Ks>','Mact<','M/H>','M/H<'])
    corr = [-0.5, -2.5, 15., 6., 1.5, -.5, .5]
    corrections.loc[0] = corr
    corrections.to_csv('../Output/data_selection.csv')

    #Set selection criteria
    df = df[df.M_ks < corr[0]]
    df = df[df.M_ks > corr[1]]

    df = df[df.Ks < corr[2]]
    df = df[df.Ks > corr[3]]

    df = df[df.Mact < corr[4]]
    df = df[df.Mact > 1.]

    df = df[df['[M/H]'] > corr[5]]
    df = df[df['[M/H]'] < corr[6]]

    # df = df[df.logg < 2.6]
    # df = df[df.logg > 2.475]
    # df = df[df.logL < 3.2]

    df = df[df['Ks'] < 10.5]
    df = df[df['Ks'] > 7.0]

    # df = df[df.stage == 4]
    df = df[0:5000]

    df = get_errors(df)

    return df.M_ks.values, df.Ks.values, df.stage, df

def get_errors(df):
    DR = 2  #Choosing the data release
    if DR == 1:
        df['pi_err'] = np.abs(0.3 + np.random.normal(0,0.5,len(df)) * 0.3) #mas

    if DR == 2:
        df['pi_err'] = np.abs(10.e-3 + np.random.normal(0,0.5,len(df)) * 2.e-3) #mas

    df['d'] = 10**(1+ (df['m-M0']/5))       #Getting all distances (pc)
    df['pi'] = 1000/df['d']                 #Getting all parallax (mas)
    df['sig_d'] = (1000*df['pi_err']/df['pi']**2)  #Propagating the Gaia error

    df['sig_mu'] = np.sqrt( (5*np.log10(np.e)/df['d'])**2 * df['sig_d']**2 )
    df['sig_M'] = df['sig_mu']  #Until we incorporate errors on Aks and Ks

    # plt.errorbar(df['pi'],df['Ks'],xerr=df['pi_err'],alpha=.1,fmt=".k",c='grey',zorder=999)
    # plt.scatter(df['pi'],df['Ks'],s=5,zorder=1000)
    # plt.show()
    #
    # plt.errorbar(df['d'],df['Ks'],xerr=df['sig_d'],alpha=.1,fmt=".k",c='grey',zorder=999)
    # plt.scatter(df['d'],df['Ks'],s=5,zorder=1000)
    # plt.show()
    return df



if __name__ == '__main__':
    plt.close('all')
    sfile = glob.glob('../Cuts_Data/cuts_MH_JKs_logg.txt')[0]
    # sfile = glob.glob('../data/Ben_Fun/TRI3*')[0]
    # sfile = glob.glob('../data/TRILEGAL_sim/*all*.txt')[0]o
    odf = pd.read_csv(sfile, sep='\s+')

    labels = odf.stage
    cheb = (labels==4)# & (odf.logAge > 9.4)
    df = odf[:]

    numplots = 5
    fig, ax = plt.subplots(numplots,2,sharex=True)
    i = 0
    for lo, hi in zip(np.linspace(-0.5,.5-1./numplots,numplots),np.linspace(-.5+1./numplots,.5,numplots)):
        tdf = df[(df['[M/H]'] > lo) & (df['[M/H]'] < hi)]
        ax[i,0].hist(tdf.M_ks,histtype='step',bins='sqrt')
        ax[i,0].set_title('Ks: '+str(np.round(lo,2))+r"$<$[M/H]$<$"+str(np.round(hi,2)))
        ax[i,1].hist(tdf.M_j,histtype='step',bins='sqrt')
        ax[i,1].set_title('J: '+str(np.round(lo,2))+r"$<$[M/H]$<$"+str(np.round(hi,2)))

        i += 1
    fig.tight_layout()
    plt.show()

    rgb = (labels!=4)

    '''Correct data'''
    odf['Aks'] = 0.114*odf.Av #Cardelli+1989>-
    odf['M_ks'] = odf.Ks - odf['m-M0'] - odf.Aks
    odf['Aj'] = 0.282*odf.Av
    odf['M_j'] = odf.J - odf['m-M0'] - odf.Aj
    odf['JKs'] = odf.J - odf.Ks
    odf = odf[odf.M_ks < -1.4]
    odf = odf[odf.M_ks > -2.0]
    # odf = odf[odf.Mact < 1.5]
    df = odf[:]

    fig, ax = plt.subplots()
    # plt.scatter(df.logTe,df.logL,s=3,c=df['[M/H]'])
    # plt.scatter(df.logTe,df.logL,s=3,c=df.Mact)
    a1 = ax.scatter(df.logTe,df.logL,s=3,c=df.Mact)
    fig.gca().invert_xaxis()
    fig.colorbar(a1)

    fig2, ax2 = plt.subplots()
    # plt.scatter(df.logTe,df.logL,s=3,c=df['[M/H]'])
    # plt.scatter(df.logTe,df.logL,s=3,c=df.Mact)
    a2 = ax2.scatter(df.M_ks,df.Ks,s=3,c=df.Mact)
    fig2.colorbar(a2)

    bins = int(np.sqrt(len(df.M_ks)))
    hig, hax = plt.subplots(3,sharex=True)
    hax[0].hist(df.M_ks,histtype='step',bins=bins)
    hax[1].hist(df.M_ks[cheb],histtype='step',bins=int(np.sqrt(len(df.M_ks[cheb]))))
    hax[2].hist(df.M_ks[rgb],histtype='step',bins=int(np.sqrt(len(df.M_ks[rgb]))))
    cp.show()



    fig, ax = plt.subplots()
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    label = ['Pre-Main Sequence', 'Main Sequence', 'Subgiant Branch', 'Red Giant Branch', 'Core Helium Burning',\
                '??', '??', 'Asymptotic Giant Branch','??']
    for i in range(len(c)):
        ax.scatter(df.M_ks[labels==i],df.Ks[labels==i],s=1,c=c[i],label=label[i])
    ax.set_ylabel('Apparent Mag in Ks')
    ax.set_xlabel('Absolute Mag in Ks')
    plt.show()

    fig, ax = plt.subplots(2)
    for i in range(len(c)):
        ax[0].scatter(df.M_ks[labels==i],df['[M/H]'][labels==i],s=1,c=c[i],label=label[i])
    ax[0].set_ylabel('[M/H]')
    ax[0].set_xlabel('Absolute Mag in Ks')
    ax[1].hist(df['[M/H]'], histtype='step',bins=int(np.sqrt(len(df.Ks))))
    ax[1].set_xlabel('[M/H]')
    plt.show()

    fig, ax = plt.subplots(2)
    for i in range(len(c)):
        ax[0].scatter(df.M_ks[labels==i],df.logg[labels==i],s=1,c=c[i],label=label[i])
    ax[0].set_ylabel('logg')
    ax[0].set_xlabel('Absolute Mag in Ks')
    ax[1].hist(df.logg, histtype='step',bins=int(np.sqrt(len(df.Ks))))
    ax[1].set_xlabel('logg')
    plt.show()

    fig, ax = plt.subplots(2)
    for i in range(len(c)):
        ax[0].scatter(df.M_ks[labels==i],df.logTe[labels==i],s=1,c=c[i],label=label[i])
    ax[0].set_ylabel('LogTeff')
    ax[0].set_xlabel('Absolute Mag in Ks')
    ax[1].hist(df.logTe, histtype='step',bins=int(np.sqrt(len(df.Ks))))
    ax[1].set_xlabel('LogTeff')
    plt.show()

    fig, ax = plt.subplots(2)
    for i in range(len(c)):
        ax[0].scatter(df.M_ks[labels==i],df.JKs[labels==i],s=1,c=c[i],label=label[i])
    ax[0].set_ylabel('J-Ks')
    ax[0].set_xlabel('Absolute Mag in Ks')
    ax[1].hist(df.JKs, histtype='step',bins=int(np.sqrt(len(df.Ks))))
    ax[1].set_xlabel('J-Ks')
    plt.show()

    fn = np.polyfit(df.M_ks, df.M_j, 1)
    fy = df.M_ks*fn[0] + fn[1]
    fig, ax = plt.subplots(2)
    for i in range(len(c)):
        ax[0].scatter(df.M_ks[labels==i],df.M_j[labels==i],s=1,c=c[i],label=label[i])
    ax[0].set_ylabel('Absolute Mag in J')
    ax[0].set_xlabel('Absolute Mag in Ks')
    ax[1].hist(df.M_ks, histtype='step',bins=int(np.sqrt(len(df.Ks))))
    ax[1].set_xlabel('J-Ks')

    ax[0].plot(df.M_ks, fy,c='r')

    plt.show()


    jfy = df.M_j - df.M_ks
    '''Getting the FG Model'''
    n, b = np.histogram(jfy[labels==4],bins=int(np.sqrt(len(labels==4))))
    mu = b[np.argmax(n)]
    sig = np.std(jfy[labels==4])
    gauss_x = np.exp(-0.5 * (((mu - jfy) / sig)**2 + 2*np.log(sig) +np.log(2*np.pi)))

    '''Getting the BG Model'''
    n, b = np.histogram(jfy[labels!=4],bins=int(np.sqrt(len(labels!=4))))
    x0 = b[np.argmax(n)]
    gamma = 0.04
    lor_x = np.exp(2*np.log(gamma) - np.log(np.pi*gamma) - np.log((jfy-x0)**2 + gamma**2))

    fig, ax = plt.subplots(2,2)
    for i in range(len(c)):
        ax[0,0].scatter(jfy[labels==i],df.M_ks[labels==i],s=1,c=c[i],label=label[i])
    ax[0,0].set_xlabel('Absolute Mag in J - Absolute Mag in Ks')
    ax[0,0].set_ylabel('Absolute Mag in Ks')
    ax[0,1].hist(jfy, histtype='step',bins=int(np.sqrt(len(jfy))),normed=True)
    ax[0,1].set_xlabel('J - K (RGB+CHeB+Others)')
    ax[1,0].hist(jfy[labels==4], histtype='step',bins=int(np.sqrt(len(labels==4))),normed=True)
    ax[1,0].set_xlabel('J - K for CHeB')
    ax[1,1].hist(jfy[(labels!=4)], histtype='step',bins=int(np.sqrt(len((labels!=4)))),normed=True)
    ax[1,1].set_xlabel('J - K for RGB+Others')

    ax[1,0].scatter(jfy,gauss_x,c='blue',s=1)
    ax[1,1].scatter(jfy,lor_x,c='orange',s=1)
    ax01 = ax[0,1].twinx()
    ax01.scatter(jfy,gauss_x + lor_x,c='green',s=1)

    fig.tight_layout()

    plt.show()





    sys.exit()
    x, y, labels, df = get_values()

    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.hist(x, histtype='step',bins=int(np.sqrt(len(x))),normed=True)

    for gamma in (np.arange(0.01,0.1,0.01)):
        x0 = -1.67
        lor = (1/np.pi*gamma) * ( gamma**2 / ((x-x0)**2 + gamma**2))

        lnlor = 2*np.log(gamma) - np.log(np.pi*gamma) - np.log((x-x0)**2 + gamma**2)

        ax2.scatter(x, np.exp(lnlor), s=5)
    plt.show()





    sys.exit()


    # vals = [-1.63,-1.59,-1.627,-1.626]
    # errs = [0.002,0.005,0.20,0.057]
    # noms = ['MixMod 2d', 'Mixmod 3d', 'RANSAC', 'Hawkins+17']
    # x = np.arange(2,10,2)
    # # fig, ax = plt.subplots()
    # # ax.errorbar(x,vals,yerr=errs, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    # # ax.scatter(x,vals,s=5,c='w',zorder=1000)
    # # ax.set_xticks(x)
    # # ax.set_xticklabels(noms,fontsize=10)
    # # ax.set_ylabel('Absolute Red Clump magnitude (TRILEGAL)')
    # # fig.tight_layout()
    # # plt.show()

    x, y, labels, df = get_values()
    # xerr = df['sig_M']

    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    label = ['Pre-Main Sequence', 'Main Sequence', 'Subgiant Branch', 'Red Giant Branch', 'Core Helium Burning',\
                'RR Lyrae variables', 'Cepheid Variables', 'Asymptotic Giant Branch','Supergiants']
    # for i in range(int(np.nanmax(labels))+1):
    #     ax.scatter(x[labels==i], y[labels==i], c=c[i], s=1,zorder=1000,label=label[i])
    # ax.scatter(x[labels==3], y[labels==3], c=c[3], s=3,zorder=1000,label=label[3])
    # ax.scatter(x[labels==4], y[labels==4], c=c[4], s=3,zorder=1000,label=label[4])

    fig, ax = plt.subplots(2,2,sharex=True,sharey=True)
    aM = ax[0,0].scatter(x,y,cmap='viridis',c=df.Mact,s=3,vmin=0.,vmax=1.5)
    aZ = ax[0,1].scatter(x,y,cmap='BrBG',c=df['[M/H]'],s=3)
    aG = ax[1,0].scatter(x,y,cmap='RdBu',c=df.logg,s=3,vmin=2.475,vmax=2.6)
    aA = ax[1,1].scatter(x,y,cmap='RdBu_r',c=df.logAge,s=3)

    ax[0,0].set_xlabel(r"$M_{Ks}$")
    ax[0,0].set_ylabel(r"$m_{Ks}$")
    ax[0,0].grid()
    ax[1,0].grid()
    ax[0,1].grid()
    ax[1,1].grid()
    ax[0,0].set_axisbelow(True)
    ax[1,0].set_axisbelow(True)
    ax[0,1].set_axisbelow(True)
    ax[1,1].set_axisbelow(True)

    fig.colorbar(aM,label='Mass',extend='both',ax=ax[0,0])
    fig.colorbar(aZ,label='[M/H]',extend='both',ax=ax[0,1])
    fig.colorbar(aG,label='log(g)',extend='both',ax=ax[1,0])
    fig.colorbar(aA,label='log(Age)',extend='both',ax=ax[1,1])
    fig.suptitle(r"Labeled TRILEGAL simulated data for (M $<$ 1.5Msol, -.5 $<$ [M/H] $<$ .5)")

    fig2, ax2 = plt.subplots()
    for i in range(int(np.nanmax(labels))+1):
        ax2.scatter(x[df.stage==i],y[df.stage==i],s=3,c=c[i])

    plt.show()
    plt.close('all')

    sys.exit()




    # sfile = glob.glob('../data/AM_TRI/k1.7*.txt')[0]
    # sfile = glob.glob('../data/AM_TRI/k1.7*.txt')[0]
    sfile = glob.glob('../data/TRILEGAL_sim/*all*.txt')[0]
    df = pd.read_csv(sfile, sep='\s+')

    # corr = [-0.5, -2.5, 15., 6., 1.4, -.5, .5]
    # df['Aks'] = 0.114*df.Av #Cardelli+1989>-
    # df['M_ks'] = df.Ks - df['m-M0'] - df.Aks
    #
    #
    # #Set selection criteria
    # df = df[df.M_ks < corr[0]]
    # df = df[df.M_ks > corr[1]]
    #
    # df = df[df.Ks < corr[2]]
    # df = df[df.Ks > corr[3]]

    # df = df[df.Mact < corr[4]]

    # df = df[df['[M/H]'] > corr[5]]
    # df = df[df['[M/H]'] < corr[6]]

    # df = df[df.stage < 5]

    '''This function corrects for extinction and sets the RC search range'''
    m_ks = df['Ks'].values
    mu = df['m-M0'].values
    Av = df['Av'].values
    M = df['Mact'].values
    labels = df['stage'].values
    Zish = df['[M/H]'].values
    logT = df['logTe'].values
    logL = df['logL'].values

    fig, ax = plt.subplots()
    fig2, ax2 = plt.subplots(3,3)
    c = ['r','b','c','g','y','k','m','darkorange','chartreuse']
    label = ['Pre-Main Sequence', 'Main Sequence', 'Subgiant Branch', 'Red Giant Branch', 'Core Helium Burning',\
                '??', '??', 'Asymptotic Giant Branch','Supergiants']
    loc = [(0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)]
    for i in range(int(np.nanmax(labels))+1):
        # if i != 5:
        #     if i != 6:
        #         if i!= 8:
        ax.scatter(logT[labels==i],logL[labels==i],s=5,c=c[i],alpha=.3,label=label[i])
        im = ax2[loc[i]].scatter(logT[labels==i],logL[labels==i],s=10,c=M[labels==i],\
                            cmap='viridis',vmin=0.0,vmax=2.)
        ax2[loc[i]].set_title(str(i))
        ax2[loc[i]].invert_xaxis()

    ax.scatter(logT[labels==4],logL[labels==4],s=5,c=c[4],alpha=.5)
    ax.legend(loc='best',fancybox=True)
    ax.invert_xaxis()
    ax.set_xlabel(r"$log_{10}(T_{eff})$")
    ax.set_ylabel(r'$log_{10}(L)$')
    ax.set_title(r"HR Diagram for a TRILEGAL dataset of the $\textit{Kepler}$ field")
    ax.grid()
    ax.set_axisbelow(True)
    fig2.tight_layout()

    fig2.subplots_adjust(right=0.8)
    cbar_ax = fig2.add_axes([0.85,0.15,0.05,0.7])
    fig2.colorbar(im,cax=cbar_ax)

    # fig.tight_layout()
    # fig.savefig('/home/oliver/Dropbox/Papers/Midterm/Images/C4_HR.png')
    plt.show(fig)
    plt.close('all')


    '''Comparisonplot'''
    sfile = glob.glob('../data/TRILEGAL_sim/*all*.txt')[0]
    dff = pd.read_csv(sfile, sep=',')
    dff['Aks'] = 0.114*dff.Av #Cardelli+1989>-
    dff['M_ks'] = dff.Ks - dff['m-M0'] - dff.Aks
    corr = [-1.0, -2.5, 15., 6., 1.5, -.5, .5]

    #Set selection criteria
    df = dff[dff.M_ks < corr[0]]
    df = df[df.M_ks > corr[1]]

    df = df[df.Ks < corr[2]]
    df = df[df.Ks > corr[3]]

    df = df[df.Mact < corr[4]]

    df = df[df['[M/H]'] > corr[5]]
    df = df[df['[M/H]'] < corr[6]]
    # df = df[df.logL < 3.2]
    # df = dff[(dff.stage == 4) | (dff.stage == 5) | (dff.stage==6)]

    m_ks = df['Ks'].values
    mu = df['m-M0'].values
    Av = df['Av'].values
    M = df['Mact'].values
    labels = df['stage'].values
    Zish = df['[M/H]'].values
    logL = df['logL'].values
    logT = df['logTe'].values
    logg = df['logg'].values

    c=[]
    for i, row in enumerate(labels):
        if labels[i]==4:
            c.append('r')
        if labels[i]==5:
            c.append('b')
        if labels[i]==6:
            c.append('k')
        if labels[i]==3:
            c.append('c')

    '''Comp-plot'''
    for i in range(1):
        fig = plt.figure(figsize=(8,4))
        ax1 = fig.add_axes([0.1,0.2,0.35,0.6])
        ax2 = fig.add_axes([0.55,0.2,0.35,0.6],sharex=ax1,sharey=ax1)

        s1 = ax1.scatter(logT,logL,s=10,c=M,\
                    cmap='viridis',vmin=0.7,vmax=1.5)
        fig.colorbar(s1,label='Mass ($M_\odot$)',ax=ax1)
        ax1.set_xlabel(r"$log_{10}(T_{eff})$")
        ax1.set_ylabel(r"$log_{10}(L)$")
        ax1.set_title('CHeB stars coloured by Mass')
        ax1.invert_xaxis()

        cmap = plt.cm.get_cmap('coolwarm')
        s2 = ax2.scatter(logT,logL,s=10,c=logg,\
                    cmap=cmap)#, vmin=-0.5, vmax=0.5)
        fig.colorbar(s2,label='logg',extend='min',ax=ax2)
        cmap.set_under("b")
        ax2.set_xlabel("$log_{10}(T_{eff})$")
        ax2.set_ylabel("$log_{10}(L)$")
        ax2.set_title('CHeB stars coloured by log(g)')

        # fig.savefig('/home/oliver/Dropbox/Papers/Midterm/Images/C4_CHeB.png')
        plt.show()
        plt.close('all')


    sys.exit()
    x, y = get_values()
    xerr = np.abs(0.01 + np.random.normal(0, 1, len(x)) * 0.005)
    yerr = np.abs(0.1 + np.random.normal(0, 1, len(y)) * 0.05)


    start_params = np.array([-1.6, 0.2, 0.5, -2.0, 1.0])
    b = -1.6
    sigrc = 0.05
    o = 1.0
    sigo = 1.0

    sig1 = np.sqrt(sigrc**2 + xerr**2)
    sig2 = np.sqrt(sigo**2 + xerr**2)

    '''Trying out new likelihoods'''
    lrc = -0.5 * (x - b)**2 / sig1**2 - np.log(sig1)

    xn = np.abs(x)
    bn = np.abs(b)
    lbg = -np.log(xn) - np.log(sig2) - 0.5 * (np.log(xn) - bn)**2/sig2**2

    # plt.scatter(x,np.exp(lrc), c='r')
    plt.scatter(x,lbg, c='c')
    plt.show()


    sys.exit()

    '''Priors for RC data run'''
    #m, b, Q, M, V
    bounds = [(-1.8,-1.4), (0, 1), (10.,13.), (0.1, 4.)]
    p0 = np.array([-1.6, 0.5, 11.0, 2.0])
    def lnprior(p):
        # We'll just put reasonable uniform priors on all the parameters.
        if not all(b[0] < v < b[1] for v, b in zip(p, bounds)):
            return -np.inf
        return 0

    # The "foreground" linear likelihood:
    def lnlike_fg(p):
        b, _, M, V = p
        model = np.zeros(y.shape) + b
        return -0.5 * ((model - x) / xerr)**2 - np.log(xerr)

    # The "background" outlier likelihood:
    def lnlike_bg(p):
        _, Q, M, V = p
        return -0.5 * ((M - y)/ V)**2 - np.log(V)

    # Full probabilistic model.
    def lnprob(p):
        b, Q, M, V = p

        # First check the prior.
        lp = lnprior(p)
        if not np.isfinite(lp):
            return -np.inf, None

        # Compute the vector of foreground likelihoods and include the q prior.
        ll_fg = lnlike_fg(p)
        arg1 = ll_fg + np.log(Q)

        # Compute the vector of background likelihoods and include the q prior.
        ll_bg = lnlike_bg(p)
        arg2 = ll_bg + np.log(1.0 - Q)

        # Combine these using log-add-exp for numerical stability.
        ll = np.sum(np.logaddexp(arg1, arg2))

        # We're using emcee's "blobs" feature in order to keep track of the
        # foreground and background likelihoods for reasons that will become
        # clear soon.
        return lp + ll, (arg1, arg2)

    # Initialize the walkers at a reasonable location.
    ndim, nwalkers = 4, 32
    p0 = [p0 + 1e-5 * np.random.randn(ndim) for k in range(nwalkers)]

    # Set up the sampler.
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)

    # Run a burn-in chain and save the final location.
    pos, _, _, _ = sampler.run_mcmc(p0, 500)

    # Run the production chain.
    sampler.reset()
    sampler.run_mcmc(pos, 1500);

    labels = ["$b$", "$Q$", "$M$", "$V$"]
    # truths = true_params + [true_frac, true_outliers[0], true_outliers[1]]
    corner.corner(sampler.flatchain, bins=35, labels=labels)#, truths=truths);

    plt.show()

    '''Doing posteriors'''
    norm = 0.0
    post_prob = np.zeros(len(x))
    for i in range(sampler.chain.shape[1]):
        for j in range(sampler.chain.shape[0]):
            ll_fg, ll_bg = sampler.blobs[i][j]
            post_prob += np.exp(ll_fg - np.logaddexp(ll_fg, ll_bg))
            norm += 1
    post_prob /= norm

    rc = np.median(sampler.flatchain[:,0])
    err = np.std(sampler.flatchain[:,0])
    rcy = np.linspace(6,15,10)

    rcx = np.linspace(rc-err,rc+err,10)
    rcy1 = np.ones(rcx.shape) * 15
    rcy2 = np.ones(rcx.shape) * 7

    # Plot the data points.
    plt.errorbar(x, y, xerr=xerr, fmt=",k", ms=0, capsize=0, lw=1, zorder=999)
    # Plot the (true) outliers.
    plt.scatter(x, y, marker="s", s=22, c=post_prob, cmap="Blues_r", vmin=0, vmax=1, zorder=1000)
    plt.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001)
    plt.title('Clump luminosity: '+str(rc))
    plt.ylim([6,15])

    # Plot the (true) good points.
    # plt.scatter(x[~m_bkg], y[~m_bkg], marker="o", s=22, c=post_prob[~m_bkg], cmap="Blues_r", vmin=0, vmax=1, zorder=1000)

    # Plot the true line.
    # plt.plot(x0, y0, color="k", lw=1.5)

    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.show()

    plt.scatter(x, y, c=post_prob, cmap="Blues_r", s=4,zorder=1000)
    plt.fill_between(rcx,rcy1,rcy2,color='red',alpha=0.8,zorder=1001)
    # ax.fill_between(fraclong, lolong, uplong,interpolate=True, facecolor='cyan')
    # plt.plot(rcx,rcy,lw=2,c='r',zorder=1001)
    # plt.axvline(rc+err,linestyle='--')
    # plt.axvline(rc-err,linestyle='--')
    plt.title('Clump luminosity: '+str(rc))
    plt.ylim([6,15])
    plt.show()
