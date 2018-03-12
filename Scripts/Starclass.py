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


class Star:
    '''
    An object class that stores and calls all astrophysical components
    necessary for finding a targets absolute magnitude in a given band.

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
        if self.band == None:
            print('You did not pass a band string when passing magnitude.\nThe band has been set to Ks.')
        if self.frame == None:
            print('You did not pass a frame string when passing position.\nThe frame has been set to ICRS.')
        return kill

    def pass_parallax(self, par, err=None):
        #Give a sanity check that we're dealing with mas
        if (par < 0.01) or (par > 5.):
            print(r'Are you 100% certain that parallax is in units of milliarcsec?')
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

    def get_A(self):
        #All values from Green+18
        coeffs = pd.DataFrame({'g':3.384,'r':[2.483],'i':[1.838],'z':[1.414],'y':[1.126],'J':[0.650],'H':[0.327],'Ks':[0.161]})
        #Check we have the conversion for this band of magnitudes
        if not any(self.band in head for head in list(coeffs)):
            print('The class cant handle this specific band yet.')
            print('Please double check youve input the string correctly. The list of possible bands is:')
            print(list(coeffs))
            print('And you passed in: '+self.band)
            sys.exit()

        #Call the Bayestar catalogue
        bayestar = BayestarWebQuery(version='bayestar2017')
        coords = SkyCoord(self.ra*units.deg, self.dec*units.deg,
                distance=(1000/self.oo)*units.pc,frame=self.frame)
        #Find the extinction coefficient
        try:
            Av = bayestar(coords, mode='median')
        except HTTPError:
            print('No Av values for this star. Set Av to 0. for now.')
            Av = 0.
        return Av * coeffs[self.band]

    def get_error(self):
        #Case 0: No errors
        if (self.m_err == None) & (self.oo_err == None):
            print('No errors given, error on M set to 0.')
            M_err = 0.

        #Case 1: Parallax error only
        elif self.m_err == None:
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
        A = self.get_A()      #Call in the extinction coefficient

        M_err = self.get_error()   #Propagate the error through

        return m - mu0 - A, M_err

if __name__ == '__main__':
    sfile = glob.glob('../../data/Ben_Fun/TRI3*')[0]
    columns=['Av','Jmag','Kmag','Hmag','L','Glon','Glat','dist','mu0']
    df = pd.read_csv(sfile, usecols=columns)

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
