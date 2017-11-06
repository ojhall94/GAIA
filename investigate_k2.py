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
    return df



if __name__ == '__main__':

    files = glob.glob('../data/Ben_Fun/*3*')
    dfC3 = pd.read_csv(files[0], sep=',')
    dfT3 = pd.read_csv(files[1], sep=',')
    files = glob.glob('../data/Ben_Fun/*6*')
    dfC6 = pd.read_csv(files[0], sep=',')
    dfT6 = pd.read_csv(files[1],sep=',')

    dfC = pd.concat([dfC3,dfC6])
    dfT = pd.concat([dfT3,dfT6])

    dfC.rename(columns={'Ks' : 'M_ks', 'J' : 'M_j', 'H' : 'M_h'} , inplace=True)

    #Cardelli+1989
    Jcorr = 0.282
    Hcorr = 0.190
    Kcorr = 0.114

    #Finding K2 Apparent in Ks, TRILEGAL absolute in Ks
    dfC['Aks'] = Kcorr*dfC.Av
    dfC['Ks'] = dfC.M_ks + dfC['mu0'] + dfC.Aks
    dfT['Aks'] = Kcorr*dfT.Av
    dfT['M_ks'] = dfT.Kmag - dfT['mu0'] - dfT.Aks

    #Finding K2 Apparent in J, TRILEGAL absolute in J
    dfC['Aj'] = Jcorr*dfC.Av
    dfC['J'] = dfC.M_j + dfC['mu0'] + dfC.Aj
    dfT['Aj'] = Jcorr*dfT.Av
    dfT['M_j'] = dfT.Jmag - dfT['mu0'] - dfT.Aj

    #Finding K2 Apparent in H, TRILEGAL absolute in H
    dfC['Ah'] = Hcorr*dfC.Av
    dfC['H'] = dfC.M_h + dfC['mu0'] + dfC.Ah
    dfT['Ah'] = Hcorr*dfT.Av
    dfT['M_h'] = dfT.Hmag - dfT['mu0'] - dfT.Ah

    #Calculing HR Diagram values
    dfC['L'] = 4*np.pi*(dfC.rad*695700e3)**2*5.67e-8*dfC.Teff**4
    dfC['logL'] = np.log10(dfC['L']/3.828e26)
    dfC['logT'] = np.log10(dfC.Teff)


    '''MKs vs mKs comparison'''
    plt.close('all')
    fig1 = plt.figure(figsize=(8,4))
    ax11 = fig1.add_axes([0.1,0.2,0.35,0.6])
    ax12 = fig1.add_axes([0.55,0.2,0.35,0.6],sharex=ax11,sharey=ax11)
    ax11.scatter(dfC.M_ks,dfC.logL,s=3)
    ax11.scatter(dfT[dfT.label==3].M_ks,dfT[dfT.label==3].logL,alpha=.1,s=3,c='g',label='RGB')
    ax11.scatter(dfT[dfT.label==4].M_ks,dfT[dfT.label==4].logL,alpha=.1,s=3,c='y',label='CHeB')
    ax11.set_title('K2 Campaign 3 data')
    ax11.set_xlabel('Absolute Magnitude (Ks)')
    ax11.set_ylabel(r'$log_{10}(L)$')

    ax12.scatter(dfC.M_ks,dfC.logL,alpha=.1,s=3, label='C3')
    ax12.scatter(dfT[dfT.label==3].M_ks,dfT[dfT.label==3].logL,s=3,c='g',label='RGB')
    ax12.scatter(dfT[dfT.label==4].M_ks,dfT[dfT.label==4].logL,s=3,c='y',label='CHeB')
    ax12.set_title('TRILEGAL sim of K2 C3')
    ax12.legend(loc='best',fancybox=True)


    fig2 = plt.figure(figsize=(8,4))
    ax21 = fig2.add_axes([0.1,0.2,0.35,0.6])
    ax22 = fig2.add_axes([0.55,0.2,0.35,0.6],sharex=ax21,sharey=ax21)
    ax21.scatter(dfC.dist,dfC.Ks,s=3)
    ax21.scatter(dfT[dfT.label==3].dist,dfT[dfT.label==3].Kmag,alpha=.1,s=3,c='g',label='RGB')
    ax21.scatter(dfT[dfT.label==4].dist,dfT[dfT.label==4].Kmag,alpha=.1,s=3,c='y',label='CHeB')
    ax21.set_title('K2 Campaign 3 data')
    ax21.set_xlabel('Dist(pc)')
    ax21.set_ylabel('Apparent Magnitude (Ks)')

    ax22.scatter(dfC.dist,dfC.Ks,alpha=.1,s=3)
    ax22.scatter(dfT[dfT.label==3].dist,dfT[dfT.label==3].Kmag,s=3,c='g',label='RGB')
    ax22.scatter(dfT[dfT.label==4].dist,dfT[dfT.label==4].Kmag,s=3,c='y',label='CHeB')
    ax22.set_title('TRILEGAL sim of K2')
    ax22.legend(loc='best',fancybox=True)

    fig3 = plt.figure(figsize=(8,4))
    ax31 = fig3.add_axes([0.1,0.2,0.35,0.6])
    ax32 = fig3.add_axes([0.55,0.2,0.35,0.6],sharex=ax31,sharey=ax31)

    ax31.scatter(dfC.logT,dfC.Ks,s=3)
    ax31.set_title('K2 Campaign 3 data')
    ax31.invert_xaxis()
    ax31.set_xlabel(r"$log_{10}(T_{eff})$")
    ax31.set_ylabel('Apparent Magnitude (Ks)')

    ax32.scatter(dfT[dfT.label==3].logTe,dfT[dfT.label==3].Kmag,s=3,c='g',label='RGB')
    ax32.scatter(dfT[dfT.label==4].logTe,dfT[dfT.label==4].Kmag,s=3,c='y',label='CHeB')
    ax32.set_title('TRILEGAL sim of K2 C3')
    ax32.legend(loc='best',fancybox=True)

    fig4 = plt.figure(figsize=(8,4))
    ax41 = fig4.add_axes([0.1,0.2,0.35,0.6])
    ax42 = fig4.add_axes([0.55,0.2,0.35,0.6],sharex=ax41,sharey=ax41)

    ax41.scatter(dfC.M_ks,dfC.Ks,s=3)
    ax41.set_title('K2 Campaign 3 data')
    ax41.invert_xaxis()
    ax41.set_xlabel("Aboluste Magnitude (Ks)")
    ax41.set_ylabel('Apparent Magnitude (Ks)')

    ax42.scatter(dfT[dfT.label==3].M_ks,dfT[dfT.label==3].Kmag,s=3,c='g',label='RGB')
    ax42.scatter(dfT[dfT.label==4].M_ks,dfT[dfT.label==4].Kmag,s=3,c='y',label='CHeB')
    ax42.set_title('TRILEGAL sim of K2 C3')
    ax42.legend(loc='best',fancybox=True)

    fig5, ax5 = plt.subplots(2,2,sharex=True)
    ax5[0,0].scatter(dfC.M_ks,dfC.Ks,s=3)
    ax5[0,0].set_title('K2 C3 and C6 data')
    ax5[0,0].set_xlabel("Aboluste Magnitude (Ks)")
    ax5[0,0].set_ylabel('Apparent Magnitude (Ks)')

    ax5[1,0].hist(dfC.M_ks, bins=int(np.sqrt(len(dfC.M_ks))), color='k', histtype='step', normed=1)
    ax5[1,0].set_title('Histogram in Absolute magnitude')

    ax5[0,1].scatter(dfT[dfT.label==3].M_ks,dfT[dfT.label==3].Kmag,s=3,c='g',label='RGB')
    ax5[0,1].scatter(dfT[dfT.label==4].M_ks,dfT[dfT.label==4].Kmag,s=3,c='y',label='CHeB')
    ax5[0,1].set_title('TRILEGAL sim of K2 C3 and C6')
    ax5[0,1].legend(loc='best',fancybox=True)

    ax5[1,1].hist(dfT.M_ks, bins=int(np.sqrt(len(dfT.M_ks))), color='k', histtype='step', normed=1)
    ax5[1,1].set_title('Histogram in Absolute magnitude')
    fig5.tight_layout()
    fig5.savefig('Output/investigate_k2.png')


    '''HR Comparison'''
    fig = plt.figure(figsize=(8,4))
    ax1 = fig.add_axes([0.1,0.2,0.35,0.6])
    ax2 = fig.add_axes([0.55,0.2,0.35,0.6],sharex=ax1,sharey=ax1)

    ax1.scatter(dfC.logT,dfC.logL,s=3)
    ax1.set_title('K2 Campaign 3 data')
    ax1.invert_xaxis()
    ax1.set_xlabel(r"$log_{10}(T_{eff})$")
    ax1.set_ylabel(r'$log_{10}(L)$')

    ax2.scatter(dfT[dfT.label==3].logTe,dfT[dfT.label==3].logL,s=3,c='g',label='RGB')
    ax2.scatter(dfT[dfT.label==4].logTe,dfT[dfT.label==4].logL,s=3,c='y',label='CHeB')
    ax2.set_title('TRILEGAL sim of K2 C3')
    ax2.legend(loc='best',fancybox=True)
    plt.show()

    '''CODE GRAVEYARD'''
    # fig2 = plt.figure(figsize=(8,4))
    # ax21 = fig2.add_axes([0.1,0.2,0.35,0.6])
    # ax22 = fig2.add_axes([0.55,0.2,0.35,0.6],sharex=ax21,sharey=ax21)
    # ax21.scatter(dfC3.M_j,dfC3.J,s=3)
    # ax21.set_title('K2 Campaign 3 data')
    # ax21.set_xlabel('Absolute Magnitude (J)')
    # ax21.set_ylabel('Apparent Magnitude (J)')
    #
    # ax22.scatter(dfT3[dfT3.label==3].M_j,dfT3[dfT3.label==3].Jmag,s=3,c='g',label='RGB')
    # ax22.scatter(dfT3[dfT3.label==4].M_j,dfT3[dfT3.label==4].Jmag,s=3,c='y',label='CHeB')
    # ax22.set_title('TRILEGAL sim of K2 C3')
    # ax22.legend(loc='best',fancybox=True)
    #
    # fig3 = plt.figure(figsize=(8,4))
    # ax31 = fig3.add_axes([0.1,0.2,0.35,0.6])
    # ax32 = fig3.add_axes([0.55,0.2,0.35,0.6],sharex=ax31,sharey=ax31)
    # ax31.scatter(dfC3.M_j,dfC3.J,s=3)
    # ax31.set_title('K2 Campaign 3 data')
    # ax31.set_xlabel('Absolute Magnitude (H)')
    # ax31.set_ylabel('Apparent Magnitude (H)')
    #
    # ax32.scatter(dfT3[dfT3.label==3].M_h,dfT3[dfT3.label==3].Hmag,s=3,c='g',label='RGB')
    # ax32.scatter(dfT3[dfT3.label==4].M_h,dfT3[dfT3.label==4].Hmag,s=3,c='y',label='CHeB')
    # ax32.set_title('TRILEGAL sim of K2 C3')
    # ax32.legend(loc='best',fancybox=True)
