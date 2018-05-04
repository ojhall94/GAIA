import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from astropy.table import Table
from astropy.io import fits

import glob
from tqdm import tqdm
import sys
import os

import barbershop
from Starclass import Star
from inv_bol import Astero_Clump

#Define solar parameters
Rsol = 695700e3 #meters
Tsol = 5778 #K
Msol = 1.989e30 #Kg
Numaxsol = 3090 #Huber et al 2011ish
Dnusol = 135.1
stefboltz = 5.670367e-8 #Wm-2K-4
Mbolsol = 4.74  #Torres
Lsol = 4 * np.pi * stefboltz * Rsol**2 * Tsol**4
loggsol = np.log10(274)

def get_yu():
    #Read in Jie Yu
    print('Reading in Yu+2018')
    sfile = '/home/oliver/PhD/Catalogues/RC_catalogues/Yu+18_table1.txt'
    yu18_1 = pd.read_csv(sfile, sep='|')
    sfile = '/home/oliver/PhD/Catalogues/RC_catalogues/Yu+18_table2.txt'
    yu18_2 = pd.read_csv(sfile, sep='|')
    yu18 = pd.merge(yu18_1, yu18_2, on='KICID',how='outer')
    yu18.rename(columns={'KICID':'KIC','EvoPhase':'stage'},inplace=True) #For consistency
    yu18.stage[yu18.stage == 0] = 'U'
    yu18.stage[yu18.stage == 1] = 'RGB'
    yu18.stage[yu18.stage == 2] = 'HeB'

    #Lets build our new dataframe!
    df = pd.DataFrame()
    HeB = yu18.stage == 'HeB'
    df['KIC'] = yu18.KIC[HeB]
    df['numax'] = yu18.numax[HeB]
    df['numax_err'] = yu18['err_x'][HeB]
    df['dnu'] = yu18.dnu
    df['dnu_err'] = yu18['err.1_x'][HeB]
    df['Teff'] = yu18.Teff[HeB]
    df['Teff_err'] = yu18['err_y'][HeB]
    df['[Fe/H]'] = yu18['Fe/H'][HeB]
    df['[Fe/H]_err'] = yu18['err.2_y'][HeB]
    df['logg'] = yu18['logg'][HeB]
    df['logg_err'] = yu18['err.1_y'][HeB]
    df['M'] = yu18['M_noCorrection'][HeB]
    df['M_err'] = yu18['err.3_y'][HeB]
    df['R'] = yu18['R_noCorrection'][HeB]
    df['R_err'] = yu18['err.4_y'][HeB]
    return df

def get_apogee():
    #The APOGEE columns we want are:
    print('Reading in APOGEE...')
    columns = ('APOGEE_ID', 'J','J_ERR','H','H_ERR','K','K_ERR','RA','DEC','GLON','GLAT',\
                'TEFF','TEFF_ERR','LOGG','LOGG_ERR','M_H','M_H_ERR','ALPHA_M','ALPHA_M_ERR')
    sfile = '/home/oliver/PhD/Catalogues/APOGEE/APOGEE_DR14.fits'
    infile = fits.open(sfile)
    fdata = infile[1].data
    adf = pd.DataFrame()
    #Build a new array with only the data I want
    for column in columns:
        adf[column] = fdata[column]
    del infile
    del fdata
    return adf


if __name__ == "__main__":
    '''Asteroseismology tester run'''
    #Read in Yu
    df = get_yu()

    AC = Astero_Clump(df, df.numax, df.dnu, df.Teff,\
        _numax_err = df.numax_err, _dnu_err = df.dnu_err, _Teff_err = df.Teff_err)
    df['ast_R'] = AC.get_radius()/Rsol
    df['ast_R_err'] = AC.get_radius_err()/Rsol
    df['ast_M'] = AC.get_mass()/Msol
    df['ast_M_err'] = AC.get_mass_err()/Msol
    df['ast_logg'] = AC.get_logg()
    df['ast_logg_err'] = AC.get_logg_err()
    df['L'] = AC.get_luminosity()/Lsol
    df['L_err'] = AC.get_luminosity_err()/Lsol
    df['Mbol'] = AC.get_bolmag()
    df['Mbol_err'] = AC.get_bolmag_err()
    # df['M_Ks'] = AC.get_M(band='Ks')
    # df['M_J'] = AC.get_J(band='J')
    # df['M_H'] = AC.get_H(band='H')

    #Cut low mass stars as recommended by Yvonne
    # df = df[df.M > 0.8]
    # df = df.reindex()

    plt.scatter(df.logg,df.ast_logg,s=3,zorder=1000)
    plt.errorbar(df.logg,df.ast_logg,xerr=df.logg_err,yerr=df.ast_logg_err,c='grey',alpha=.3,fmt=None)
    plt.plot(df.logg,df.logg,c='r',alpha=.5,linestyle='--',zorder=998)
    plt.xlabel('Yu+18')
    plt.ylabel('Astero logg')
    plt.show()

    plt.scatter(df.logg_err,df.ast_logg_err,s=3,zorder=1000)
    plt.plot(df.logg_err,df.logg_err,c='r',alpha=.5,linestyle='--',zorder=998)
    plt.xlabel('Yu+18')
    plt.ylabel('Astero logg error')
    plt.show()

    sys.exit()
    plt.scatter(df.M,df.ast_M,s=3,zorder=1000)
    plt.errorbar(df.M,df.ast_M,xerr=df.M_err,yerr=df.ast_M_err,c='grey',alpha=.3,fmt=None)
    plt.plot(df.M,df.M,c='r',alpha=.5,linestyle='--',zorder=998)
    plt.xlabel('Yu+18')
    plt.ylabel('Astero mass')
    plt.show()

    plt.scatter(df.R,df.ast_R,s=3,zorder=1000)
    plt.errorbar(df.R,df.ast_R,xerr=df.R_err,yerr=df.ast_R_err,c='grey',alpha=.3,fmt=None)
    plt.plot(df.R,df.R,c='r',alpha=.5,linestyle='--',zorder=998)
    plt.xlabel('Yu+18')
    plt.ylabel('Astero radius')
    plt.show()

    plt.scatter(df.M_err,df.ast_M_err,s=3,zorder=1000)
    plt.plot(df.M_err,df.M_err,c='r',alpha=.5,linestyle='--',zorder=998)
    plt.xlabel('Yu+18')
    plt.ylabel('Astero mass error')
    plt.show()

    plt.scatter(df.R_err,df.ast_R_err,s=3,zorder=1000)
    plt.plot(df.R_err,df.R_err,c='r',alpha=.5,linestyle='--',zorder=998)
    plt.xlabel('Yu+18')
    plt.ylabel('Astero radius error')
    plt.show()


    sys.exit()

    barber = barbershop.open(df,'Mbol','numax')
    barber.histograms_on(x=True)
    barber.add_client('R')
    barber.add_client('M')
    barber.add_client('logg')
    barber.add_client('L')
    barber.add_client('Mbol')
    barber.show_mirror()
