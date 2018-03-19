import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as units
from dustmaps.bayestar import BayestarWebQuery

import glob
from tqdm import tqdm
import sys
import os

import ClosePlots as cp
import barbershop


class Star:
    '''
    An object class that stores and calls all astrophysical components
    necessary for finding a targets absolute magnitude in a given band.

    NOTE: Make sure to input numpy arrays!

    Input:
        _ID: List of star IDs for identification

    Returns:
        M (pandas dataframe): Absolute magnitude
        M_err (pandas dataframe): Error on above

    .. codeauthor:: Oliver James Hall
    '''
    def __init__(self, _ID):
        self.ID = _ID
        self.oo = 0.
        self.m = 0.
        self.ra = 9999.
        self.dec = 9999.
        self.band = None
        self.frame = None

    def check_contents(self):
        kill = False
        try:
            if self.oo <= 0.:
                print('Please pass a (positive) parallax in mas.')
                kill = True
            if self.m <= 0.:
                print('Please pass a (positive) magnitude.')
                kill = True
            if self.ra == 9999.:
                print('Please pass a RA (or appropriate longitude) in degrees.')
                kill = True
            if self.dec == 9999.:
                print('Please pass a Dec (or appropriate latitude) in degrees.')
                kill = True
        except ValueError:
            pass
        if self.band == None:
            print('You did not pass a band string when passing magnitude.\nThe band has been set to Ks.')
        if self.frame == None:
            print('You did not pass a frame string when passing position.\nThe frame has been set to ICRS.')
        return kill

    def pass_parallax(self, par, err=None):
        #Give a sanity check that we're dealing with mas
        if any(par < 0.01) or any(par > 5.):
            print(r'Are you 100% certain that (all) parallax(es) is/are in units of milliarcsec?')
        self.oo = par   #in mas
        self.oo_err = err #in mas

    def pass_position(self, ra, dec, frame='icrs'):
        self.ra = ra
        self.dec = dec
        self.frame = frame

    def pass_magnitude(self, mag, err=None, band='Ks'):
        self.m = mag
        self.m_err = err
        self.band = band

    def get_mu0(self):
        r = 1000/self.oo
        return 5*np.log10(r) - 5

    def get_Av(self):
        #Call the Bayestar catalogue
        bayestar = BayestarWebQuery(version='bayestar2017')
        coords = SkyCoord(self.ra.values*units.deg, self.dec.values*units.deg,
                distance=(1000/self.oo.values)*units.pc,frame=self.frame)
        #Find the extinction coefficient
        try:
            Av = bayestar(coords, mode='median')
        except:
            print('No Av values for this star. Set Av to 0. for now.')
            Av = 0.
        return Av

    def get_Aband(self):
        #All values from Green+18
        coeffs = pd.DataFrame({'g':3.384,'r':[2.483],'i':[1.838],'z':[1.414],'y':[1.126],'J':[0.650],'H':[0.327],'Ks':[0.161]})
        #Check we have the conversion for this band of magnitudes
        if not any(self.band in head for head in list(coeffs)):
            print('The class cant handle this specific band yet.')
            print('Please double check youve input the string correctly. The list of possible bands is:')
            print(list(coeffs))
            print('And you passed in: '+self.band)
            sys.exit()

        return self.get_Av() * coeffs[self.band].values

    def get_error(self):
        #Case 0: No errors
        try:
            if (self.m_err == None) & (self.oo_err == None):
                print('No errors given, error on M set to 0.')
                M_err = 0.

                return M_err
        except ValueError:
            pass

        #Case 1: Parallax error only
        if self.m_err == None:
                M_err = np.sqrt((5*np.log10(np.e)/self.oo)**2 * self.oo_err**2)

        #Case 2: Magnitude error only
        elif self.oo_err == None:
            M_err = self.m_err

        #Case 3: Errors on both values
        else:
            M_err = np.sqrt(self.m_err**2 + (5*np.log10(np.e)/self.oo)**2 * self.oo_err**2)

        return M_err

    def get_M(self):
        if self.check_contents():
            print('Im going to kill the run so you can pass the values correctly.')
            sys.exit()
            return None
        m = self.m       #Call in the magnitude
        mu0 = self.get_mu0()  #Call in the distance modulus
        Aband = self.get_Aband() #Call in the appropriate extinction coeff

        M_err = self.get_error()   #Propagate the error through

        return m - mu0 - Aband, M_err

if __name__ == '__main__':
    option = 0

    if option == 0:
        sfile = glob.glob('../../data/Ben_Fun/TRI3*')[0]
        # columns=['Av','Jmag','Kmag','Hmag','L','Glon','Glat','dist','mu0']
        #This file has all the info we need
        df = pd.read_csv(sfile)
        df['ID'] = range(len(df))
        df['par'] = 1000./df.dist
        df = df[df.par < 500].reindex() #Kill the weird outlier
        df['par_err'] = np.random.randn(len(df))*0.1 *df['par'] #Adding 10% errors
        df['A'] = df.Av * 0.161
        frame='galactic'
        band='Ks'
        df['Mtru'] = df.Kmag = 5*np.log10(df.dist) + 5 - df.A

        #Get absolute magnitudes
        S = Star(df.ID)
        S.pass_parallax(df.par, df.par_err)
        S.pass_position(df.Glon,df.Glat,frame)
        S.pass_magnitude(df.imag, band='i')
        AKs = S.get_Aband()
        M, M_err = S.get_M()

        df['AKs_green'] = S.get_Aband()
        df['MKs'] = M
        df['MKs_err'] = M_err

        plt.close()
        fig, ax = plt.subplots()
        col = ax.scatter(df.Mass, M,s=3,c=df.label,zorder=1000)
        ax.errorbar(df.Mass, M, c='grey',alpha=.3,fmt=None,yerr=M_err,zorder=999)
        ax.set_xlabel('Mass (Msol)')
        ax.set_ylabel('AbsMag (i)')
        fig.colorbar(col)
        ax.set_title('3 = RGB, 4 = CHeB')
        fig.gca().invert_yaxis()
        plt.show()
        sys.exit()

    if option ==1:

        sfile = glob.glob('../../data/Elsworth+/Elsworth_x_TGAS.csv')[0]
        df = pd.read_csv(sfile)
        print(list(df))

        df = df[df.M1 > 0.]             #Kill broken mass values
        #Put stage IDs into something that can be barbershopped
        df.stage[df.stage=='RGB']=0
        df.stage[df.stage=='RC']=1
        df.stage[df.stage=='2CL']=2
        df.stage[df.stage=='U'] = 3
        df.stage = df.stage.fillna(3)
        df = df.reset_index()

        length = len(df)
        #First we're going to use galactic coordinates
        M_gal = np.zeros(length)
        M_gal_err = np.zeros(length)
        for idx in tqdm(range(length)):
            S = Star(str(df.KIC[idx])) #Call class
            S.pass_parallax(df.astero_parallax[idx], err=df.astero_parallax_err[idx])
            S.pass_position(df.GLON[idx], df.GLAT[idx], frame='galactic')
            S.pass_magnitude(df.kic_imag[idx],err=None, band='i')
            M_gal[idx], M_gal_err[idx] = S.get_M()

        plt.scatter(M_gal, df.kic_imag, c=df.stage)

        df['M_i'] = M_gal
        df['M_i_err'] = M_gal_err

        barber = barbershop.open(df,'M1','M_i')
        barber.add_client('stage')
        barber.add_client('[M/H]1')
        barber.add_client('Teff')
        barber.histograms_on(x=True,y=True)
        barber.show_mirror()

        sys.exit()

    if option ==2:
        sfile = glob.glob('../../data/Ben_Fun/TRI3*')[0]
        df = pd.read_csv(sfile)

        #Now lets do the same thing but in a loop for 20 or so stars
        df['par'] = 1000./df.dist
        df['A'] = df.Av * 0.161
        frame = 'galactic'
        band = 'Ks'
        df['Mtru'] = df.Kmag - 5*np.log10(df.dist) + 5 - df.A

        length = 50
        M = np.zeros(length)
        M_err = np.zeros(length)
        for idx in tqdm(range(length)):
            S = Star(str(idx))
            S.pass_parallax(df.par[idx], err=.001)
            S.pass_position(df.Glon[idx],df.Glat[idx],frame=frame)
            S.pass_magnitude(df.Kmag[idx],err=0.,band=band)
            M[idx], M_err[idx] = S.get_M()

        fig, (ax, axr) = plt.subplots(2)
        ax.scatter(M, df.Mtru[:length],s=3,zorder=1000)
        ax.errorbar(M, df.Mtru[:length], c='grey',alpha=.3,fmt='o',xerr=M_err,zorder=999)
        ax.plot(df.Mtru, df.Mtru, c='r', alpha=.7, linestyle='--')
        ax.set_ylabel('M True')
        ax.set_xlabel('M from Class')

        axr.scatter(M,M-df.Mtru[:length],s=3,zorder=1000)
        axr.errorbar(M,M-df.Mtru[:length],c='grey',alpha=.3,fmt='o',yerr=M_err,zorder=999)
        axr.set_ylabel('M - Mtru')
        axr.set_xlabel('M from class')
        axr.axhline(0.,c='r',linestyle='--',alpha=.7)

        ax.grid()
        axr.grid()
        fig.tight_layout()
        plt.show()
