import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  #Turn off SettingWithCopyWarning
import os
import omnitool

import argparse
parser = argparse.ArgumentParser(description='Recalculate Bolometric Corrections for a given temperature perturbation')
parser.add_argument('tempdiff', default=0., type=float,
                    help='Perturbation to the temperature values in K')
parser.add_argument('stage', type=str, choices=['load','unload'],
                    help='Load prepares the data for BCcodes. Unload saves it to a location of choice.')
parser.add_argument('-pl', '--perturb_logg', action='store_const', const=True, default=False,
                    help='If true, perturb our value of logg using seismic scaling relations for the perturbed Teff')
parser.add_argument('-r', '--reddening', action='store_const', const=True, default=False,
                    help='If true, include reddening in the interpolation. WARNING: This is *not* required for the Hall+18 work.')
parser.add_argument('-f', '--flower', action='store_const', const=True, default=False,
                    help='If true, return a set of BCs calculated using the method by Flower 1996, as presented in Torres 2010')                    #  -f, --flower
args = parser.parse_args()

__datadir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/data/KepxDR2/'
__bccodes__ = os.path.expanduser('~')+'/PhD/Hacks_and_Mocks/bolometric-corrections/BCcodes/'

def get_flower(teff):
     lteff = np.log10(teff)
     BCv = np.full(len(lteff), np.nan)

     BCv[lteff<3.70] = (-0.190537291496456*10.0**5) + \
     (0.155144866764412*10.0**5*lteff[lteff<3.70]) + \
     (-0.421278819301717*10.0**4.0*lteff[lteff<3.70]**2.0) + \
     (0.381476328422343*10.0**3*lteff[lteff<3.70]**3.0)

     BCv[(3.70<lteff) & (lteff<3.90)] = (-0.370510203809015*10.0**5) + \
     (0.385672629965804*10.0**5*lteff[(3.70<lteff) & (lteff<3.90)]) + \
     (-0.150651486316025*10.0**5*lteff[(3.70<lteff) & (lteff<3.90)]**2.0) + \
     (0.261724637119416*10.0**4*lteff[(3.70<lteff) & (lteff<3.90)]**3.0) + \
     (-0.170623810323864*10.0**3*lteff[(3.70<lteff) & (lteff<3.90)]**4.0)

     BCv[lteff>3.90] = (-0.118115450538963*10.0**6) + \
     (0.137145973583929*10.0**6*lteff[lteff > 3.90]) + \
     (-0.636233812100225*10.0**5*lteff[lteff > 3.90]**2.0) + \
     (0.147412923562646*10.0**5*lteff[lteff > 3.90]**3.0) + \
     (-0.170587278406872*10.0**4*lteff[lteff > 3.90]**4.0) + \
     (0.788731721804990*10.0**2*lteff[lteff > 3.90]**5.0)

     return BCv


if __name__ == '__main__':
    if args.stage == 'load':
        cdf = pd.read_csv(__datadir__+'rcxyu18_pre_elsworth.csv')
        out = cdf[['KICID','[Fe/H]']]   #Load in fixed values
        out['Teff'] = cdf['Teff'] + args.tempdiff   #Add temperature perturbation

        if not args.perturb_logg:
            out['logg'] = cdf['logg']
        else:
            sc = omnitool.scalings(cdf.numax, cdf.dnu, out.Teff)
            out['logg'] = sc.get_logg()

        if not args.reddening:
            out['Ebv'] = np.zeros(len(out))
        else:
            out['Ebv'] = cdf['Ebv']

        out = out[['KICID','logg','[Fe/H]','Teff','Ebv']]

        out.to_csv(__bccodes__+'input.sample.all',  sep='\t', header=False, index=False,)
        print('Data loaded for Temperature perturbation of: '+str(args.tempdiff))

    if args.stage == 'unload':
        bcall = pd.read_csv(__bccodes__+'output.file.all', sep='\s+')
        bcall.rename(columns={'ID':'KICID',
                            'BC_1':'BC_J',
                            'BC_2':'BC_H',
                            'BC_3':'BC_K',
                            'BC_4':'BC_GAIA'}, inplace=True)
        bcall.drop(columns=['log(g)','[Fe/H]','Teff','E(B-V)','BC_5'], inplace=True)
        bcall.to_csv(__datadir__+'BCs/casagrande_bcs_'+str(args.tempdiff)+'_singular.csv',index=False)
