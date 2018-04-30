import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import omnitool
from omnitool.literature_values import *

import sys

def read_data():
    sfile = '/home/oliver/PhD/Gaia_Project/data/KepxDR2/xyu18.csv'
    df = pd.read_csv(sfile,index_col=0)
    return df

if __name__ == '__main__':
    df = read_data()

    #Now lets kill any values that don't have Gaia parallaxes
    print('Removing NaN parallaxes')
    print('Before: '+str(len(df)))
    df = df[np.isfinite(df.parallax)]
    df = df.reindex()
    print('After: '+str(len(df)))
    df.head(2)

    '''<<<ASTEROSEISMIC>>>'''
    #Now lets run scaling relations and get the bolometric magnitudes
    sc = omnitool.scalings(df, df.numax, df.dnu, df. Teff,\
                          _numax_err = df.numax_err, _dnu_err = df.dnu_err,\
                          _Teff_err = df.Teff_err)
    df['R'] = sc.get_radius()/Rsol
    df['R_err'] = sc.get_radius_err()/Rsol
    df['M'] = sc.get_mass()/Msol
    df['M_err'] = sc.get_mass_err()/Msol
    df['logg'] = sc.get_logg()
    df['logg_err'] = sc.get_logg_err()
    df['L'] = sc.get_luminosity()/Lsol
    df['L_err'] = sc.get_luminosity_err()/Lsol
    df['Mbol'] = sc.get_bolmag()
    df['Mbol_err'] = sc.get_bolmag_err()
    df['Z'] = Zsol * 10 ** df['[Fe/H]'].values

    #Now lets get the bolometric correction going
    get_bc = omnitool.bolometric_correction(df.Teff.values,\
                                           df.logg.values,\
                                           df.Z.values,)
    df['ast_MKs'] = df.Mbol - get_bc(band='Ks')
    df['ast_MH'] = df.Mbol - get_bc(band='H')
    df['ast_MJ'] = df.Mbol - get_bc(band='J')
    df['ast_M_err'] = np.sqrt(df.Mbol_err**2 + err_bc**2)


    '''<<<ASTROMETRIC>>>'''
    #Now lets collect the absolute magnitude from reddening
    #We're going to estimate the distance uncertainties as symmetric
    df['r_err'] = np.sqrt((df.r_hi-df.r_est)**2 + (df.r_est - df.r_lo)**2)

    sg = omnitool.spyglass()
    sg.pass_position(df.ra,df.dec,frame='icrs')
    sg.pass_distance(df.r_est, df.r_err)

    sg.pass_magnitude(df.kmag, band='Ks')
    df['phot_Mks'], df['phot_Mks_err'] = sg.get_M()
    sg.pass_magnitude(df.kmag, band='H')
    df['phot_Mh'], df['phot_Mh_err'] = sg.get_M()
    sg.pass_magnitude(df.kmag, band='J')
    df['phot_Mj'], df['phot_Mj_err'] = sg.get_M()

    '''<<<PLOTS>>>'''
