import numpy as np
import pandas as pd
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
args = parser.parse_args()

__datadir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/data/KepxDR2/'
__bccodes__ = os.path.expanduser('~')+'/PhD/Hacks_and_Mocks/bolometric-corrections/BCcodes/'

if __name__ == '__main__':
    if args.stage == 'load':
        cdf = pd.read_csv(__datadir__+'rcxyu18_pre_elsworth.csv')
        out = cdf[['KICID','[Fe/H]','Ebv']]    #Load in fixed values
        out['Teff'] = cdf['Teff'] + args.tempdiff   #Add temperature perturbation

        if not args.perturb_logg:
            out['logg'] = cdf['logg']
        else:
            sc = omnitool.scalings(cdf.numax, cdf.dnu, out.Teff)
            out['logg'] = sc.get_logg()

        out = out[['KICID','logg','[Fe/H]','Teff','Ebv']]

        out.to_csv(__bccodes__+'input.sample.all')
        print('Data loaded for Temperature perturbation of: '+str(args.tempdiff))

    if args.stage == 'unload':
        bcall = pd.read_csv(__bccodes__+'output.file.all', sep='\s+')
        bcall.rename(columns={'ID':'KICID',
                            'BC_1':'BC_K',
                            'BC_2':'BC_GAIA',
                            'BC_3':'BC_J',
                            'BC_4':'BC_H'}, inplace=True)
        bcall.drop(columns=['log(g)','[Fe/H]','Teff','E(B-V)','BC_5'], inplace=True)
        bcall.to_csv(__datadir__+'BCs/casagrande_bcs_'+str(args.tempdiff)+'.csv',index=False)
