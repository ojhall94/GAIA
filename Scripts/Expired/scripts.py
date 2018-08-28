import omnitool
from omnitool.literature_values import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import t

import warnings

'''Ive ripped this code straight from compare_magnitudes.ipynb to make it more
easily importable for use in other notebooks
'''


def get_ast_yu18():
    warnings.filterwarnings("ignore")
    #Read in Jie Yu
    sfile = '/home/oliver/PhD/Catalogues/RC_catalogues/Yu+18_table1.txt'
    cols = ['KICID','numax','err','dnu','err.1']
    yu18_1 = pd.read_csv(sfile, usecols=cols, sep='|')

    sfile = '/home/oliver/PhD/Catalogues/RC_catalogues/Yu+18_table2.txt'
    cols = ['KICID','Teff','err','Fe/H','err.2','EvoPhase']
    yu18_2 = pd.read_csv(sfile, usecols=cols, sep='|')
    yu18 = pd.merge(yu18_1, yu18_2, on='KICID',how='outer')
    yu18.rename(columns={'EvoPhase':'stage',
                        'err_x':'numax_err',
                        'err.1':'dnu_err',
                        'err_y':'Teff_err',
                         'Fe/H':'[Fe/H]',
                        'err.2':'[Fe/H]_err',
                        'err.1_y':'logg_err',
                        'err.3_y':'M_err',
                        'err.4_y':'R_err'},inplace=True) #For consistency

    #Now lets read in the KICID and relevant information from APOKASC
    sfile = '/home/oliver/PhD/Catalogues/APOKASC/APOKASC_cat_v3.6.5.txt'
    cols= ['KEPLER_ID','GAIA_PARALLAX','GAIA_PARALLAX_ERR',\
          'J_MAG_2M', 'J_MAG_ERR', 'H_MAG_2M', 'H_MAG_ERR',\
           'K_MAG_2M', 'K_MAG_ERR','RA','DEC']

    apokasc = pd.read_csv(sfile, usecols=cols, skiprows=644,sep='\s+')
    apokasc.rename(columns={'KEPLER_ID':'KICID'},inplace=True)

    #Now lets combine and make cuts
    print('X-matching Yu18 and APOKASC...')
    print('Before: '+str(len(yu18)))
    df = pd.merge(yu18, apokasc, on='KICID',how='outer').reindex()
    df = df[~df.numax.isnull()].reindex()
    print('After: '+str(len(df)))
    df.head(2)

    # #Xmatch with Gaia TGAS data
    # print('X-matching Yu18 and Gaia TGAS Data')
    # print('Before: '+str(len(df)))
    # df = df[~df.GAIA_PARALLAX.isnull()].reindex()
    # df = df[df.GAIA_PARALLAX > -9000.].reindex()
    # print('After: '+str(len(df)))

    # #Kill negative parallaxes
    # print('Removing negative parallaxes & those with error < 40%')
    # print('Before: '+str(len(df)))
    # sel = (df.GAIA_PARALLAX_ERR < 0.4*df.GAIA_PARALLAX) & (df.GAIA_PARALLAX > 0.)
    # df = df[sel].reindex()
    # print('After: '+str(len(df)))

    #Replacing negative magnitude error spaces with 0. errors
    sel = df.K_MAG_ERR <0.
    df['K_MAG_ERR'][sel] = 0.
    sel = df.H_MAG_ERR <0.
    df['H_MAG_ERR'][sel] = 0.
    sel = df.J_MAG_ERR <0.
    df['J_MAG_ERR'][sel] = 0.

    #Finally, killing any ridiculous magnitude error outliers as unrealistic
    sel = df.K_MAG_ERR >9.
    df['K_MAG_ERR'][sel] = 0.
    sel = df.H_MAG_ERR >9.
    df['H_MAG_ERR'][sel] = 0.
    sel = df.J_MAG_ERR >9.
    df['J_MAG_ERR'][sel] = 0.

    #First, lets use asteroseismology scaling relations
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

    #Now lets get the bolometric corrections
    get_bc = omnitool.bolometric_correction(df.Teff.values,\
                                           df.logg.values,\
                                           df.L.values,\
                                           df.Z.values,)
    KBC = get_bc(band='Ks')
    HBC = get_bc(band='H')
    JBC = get_bc(band='J')

    #And finally, calculate the absolute magnitudes with appropriate error
    df['ast_MKs'] = df.Mbol - KBC
    df['ast_MH'] = df.Mbol - HBC
    df['ast_MJ'] = df.Mbol - JBC
    df['ast_M_err'] = np.sqrt(df.Mbol_err**2 + err_bc**2)

    # #Now, lets use photometry
    # sg = omnitool.spyglass(str(df.KICID.values))
    # sg.pass_parallax(df.GAIA_PARALLAX, err = df.GAIA_PARALLAX_ERR)
    # sg.pass_position(df.RA, df.DEC, frame='icrs')
    #
    # #Now lets read in the various magnitude values and get out the results
    # sg.pass_magnitude(df.K_MAG_2M, err = df.K_MAG_ERR, band='Ks')
    # df['phot_MKs'], df['phot_MKs_err'] = sg.get_M()
    #
    # sg.pass_magnitude(df.H_MAG_2M, err = df.H_MAG_ERR, band='H')
    # df['phot_MH'], df['phot_MH_err'] = sg.get_M()
    #
    # sg.pass_magnitude(df.J_MAG_2M, err = df.J_MAG_ERR, band='J')
    # df['phot_MJ'], df['phot_MJ_err'] = sg.get_M()

    sns.distplot(df.ast_MKs, label='Ks')
    sns.distplot(df.ast_MH,label='H')
    sns.distplot(df.ast_MJ,label='J')
    plt.axvspan(hawkvals['Ks']-hawkerr,hawkvals['Ks']+hawkerr,alpha=.5,color='k',label='Hawkins in K,H,J')
    plt.axvspan(hawkvals['H']-hawkerr,hawkvals['H']+hawkerr,alpha=.5,color='k')
    plt.axvspan(hawkvals['J']-hawkerr,hawkvals['J']+hawkerr,alpha=.5,color='k')
    plt.title('Absolute magnitudes obtained through seismology')
    plt.xlabel('Absolute Magnitude')
    plt.legend()
    plt.show()

    # sns.distplot(df.phot_MKs, label='Ks')
    # sns.distplot(df.phot_MH,label='H')
    # sns.distplot(df.phot_MJ,label='J')
    # plt.axvspan(hawkvals['Ks']-hawkerr,hawkvals['Ks']+hawkerr,alpha=.5,color='k',label='Hawkins in K,H,J')
    # plt.axvspan(hawkvals['H']-hawkerr,hawkvals['H']+hawkerr,alpha=.5,color='k')
    # plt.axvspan(hawkvals['J']-hawkerr,hawkvals['J']+hawkerr,alpha=.5,color='k')
    # plt.title('Absolute magnitudes obtained through astrometry')
    # plt.xlabel('Absolute Magnitude')
    # plt.legend()
    # plt.show()

    return df


def normal(x, mu, sig):
    return (1/np.sqrt(2*np.pi*sig**2)) * np.exp(-(x-mu)**2/(2*sig**2))

def cauchy(x, mu, gamma):
    return 1/(np.pi*gamma*(1+((x-mu)/gamma)**2))

def students_t(x, nu, mu, sigma):
    return t.pdf(x, nu, mu, sigma)

if __name__ == "__main__":
    mu = 0.
    gamma = 0.2
    nu = 5.
    x = np.linspace(-5.,5.,1000)

    rv = t.pdf(x, nu)


    plt.plot(x, t.pdf(x,1.,0.,1.))
    plt.plot(x, t.pdf(x,1.,0.,2.))
    plt.plot(x, t.pdf(x,1.,0.,3.))

    plt.show()
