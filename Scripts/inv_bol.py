import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from scipy.interpolate import RegularGridInterpolator

import glob
from tqdm import tqdm
import sys
import h5py
import os

import ClosePlots as cp
import barbershop
from Starclass import Star

#Define solar parameters
Rsol = 695700e3 #meters
Tsol = 5778 #K
Numaxsol = 3090 #Huber et al 2011ish
Dnusol = 135.1
stefboltz = 5.670367e-8 #Wm-2K-4
Mbolsol = 4.74  #Torres
Lsol = 4 * np.pi * stefboltz * Rsol**2 * Tsol**4

class Astero_Clump:
    def __init__(self, _core_df, _numax, _dnu, _Teff, _numax_err = None, _dnu_err = None, _Teff_err = None):
        self.core_df = _core_df
        self.numax = _numax
        self.dnu = _dnu
        self.numax_err = _numax_err
        self.dnu_err = _dnu_err
        self.Teff = _Teff
        self.Teff_err = _Teff_err

    def get_radius(self):
        R = Rsol * (self.numax / Numaxsol) * (self.dnu / Dnusol)**(-2) * (self.Teff / Tsol)**(0.5)
        return R

    def get_luminosity(self):
        L = 4 * np.pi * stefboltz * self.get_radius()**2 * self.Teff**4
        return L

    def get_bolmag(self):
        Mbol = Mbolsol - 2.5*np.log10(self.get_luminosity()/Lsol)
        return Mbol

    def get_bc(self, band):
        bcmodel = h5py.File('/home/oliver/PhD/Catalogues/BCgrids/bcgrid.h5','r')#Huber+17
        bands = dict({'Ks':'bc_k','J':'bc_j','H':'bc_h','G':'bc_ga'})   #Band conversions

        interp = RegularGridInterpolator((np.array(bcmodel['teffgrid']),\
            np.array(bcmodel['logggrid']),np.array(bcmodel['fehgrid']),\
            np.array(bcmodel['avgrid'])),np.array(bcmodel[bands[band]]))

        #Gonna have to do this iteratively per star
        Av = self.get_Av()
        logg = self.core_df['log.g1']
        feh = self.core_df['kic_feh']
        BC = np.zeros(len(df))
        for idx in tqdm(range(len(df))):
            BC[idx] = interp(np.array([self.Teff[idx], logg[idx], feh[idx], Av[idx]]))
        return BC

    def get_Av(self):
        S = Star(df.KIC)
        S.pass_parallax(df.astero_parallax)
        S.pass_position(df.GLON, df.GLAT, frame='galactic')
        Av = S.get_Av()
        return Av

    def get_M(self, band='Ks'):
        BC = self.get_bc(band)
        Mabs = self.get_bolmag() - BC
        return Mabs

if __name__ == "__main__":
    #Read data
    sfile = glob.glob('../../data/Elsworth+/Elsworth_x_TGAS.csv')[0]
    odf = pd.read_csv(sfile)
    print(list(odf))

    #Kill any non-RC stars
    print(len(odf))
    df =  odf[odf.stage == 'RC']
    df = df[df.R1 > 0.]
    df = df.reset_index()
    print(len(df))

    bands = ['Ks','J','H']
    hawkvals = dict({'Ks':-1.61,'J':-0.93,'H':-1.46})
    hawkerr = 0.01
    df = df.rename(index=str, columns={'kic_kmag':'Ks','kic_jmag':'J','kic_hmag':'H'})

    for band in bands:
        #Get 'True' Absmags
        S = Star(df.KIC)
        S.pass_parallax(df.astero_parallax)
        S.pass_position(df.GLON, df.GLAT, frame='galactic')
        S.pass_magnitude(df[band],band=band)
        Mtru, _ = S.get_M()

        #Get asteroseismic absmags
        AC = Astero_Clump(df, df.numax, df.Dnu, df.Teff)
        Mast = AC.get_M(band=band)

        fig, ax = plt.subplots()
        col = ax.scatter(Mast,df[band],c=(Mtru-Mast)*100/Mtru, s=5,label='M astero')
        ax.axvspan(hawkvals[band]-0.1,hawkvals[band]+0.1,alpha=.2,color='r',label='Hawkins')
        ax.axvline(np.median(Mast),linestyle='-.',label='Median')
        ax.set_xlabel('M('+band+')')
        ax.set_ylabel('m('+band+')')
        ax.legend()
        fig.colorbar(col,label='Perc diff')
        fig.savefig('comparison_'+band+'.png')
        plt.show()

        sns.jointplot(Mast,df[band])
        plt.savefig('jointplot_'+band+'.png')
        plt.show()

    sys.exit()
    #Note, all values are returned in SI units, not solar values
    AC = Astero_Clump(df, df.numax, df.Dnu, df.Teff)
    df['Lum'] = AC.get_luminosity()
    df['logL'] = np.log10(df.Lum)
    df['logTe'] = np.log10(df.Teff)



    #Just check they keep lining up
    fig, ax = plt.subplots(2,2)
    ax[0,0].scatter(df.R2, AC.get_radius()/Rsol,zorder=1000)
    ax[0,0].plot(df.R2, df.R2,c='r',linestyle='--',zorder=999)
    ax[1,0].scatter(4 * np.pi * stefboltz * (df.R2*Rsol)**2 * df.Teff**4, AC.get_luminosity(),zorder=1000)
    ax[1,0].plot(AC.get_luminosity(), AC.get_luminosity(),c='r',linestyle='--',zorder=999)
    ax[0,1].scatter(AC.get_bolometric(),1000/df.parallax)
    plt.show()
