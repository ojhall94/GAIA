import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from astropy.coordinates import SkyCoord
import astropy.units as units
from dustmaps.bayestar import BayestarWebQuery


import sys
import os

import ClosePlots as cp


class Star:
    '''
    An object class that stores and calls all astrophysical components
    necessary for finding a targets absolute magnitude in a given band.
    .. codeauthor:: Oliver James Hall
    '''
    def __init__(self, _ID):
        self.ID = _ID
        self.oo = 0.
        self.m = 0.
        self.ra = -1.
        self.dec = -1.
        self.band = None

    def check_contents(self):
        kill = False
        if self.oo <= 0.:
            print('Please pass a (positive) parallax in mas.')
            kill = True
        if self.m <= 0.:
            print('Please pass a (positive) magnitude.')
            kill = True
        if self.ra < 0.:
            print('Please pass a positive RA in degrees.')
            kill = True
        if self.dec < 0.:
            print('Please pass a positive Dec in degrees.')
            kill = True
        if self.band == None:
            print('You did not pass a band string when passing magnitude.\nThe band has been set to Ks.')
        return kill

    def pass_parallax(self, par, err=None):
        #Give a sanity check that we're dealing with mas
        if (par < 0.01) or (par > 5.):
            print(r'Are you 100% certain that parallax is in units of milliarcsec?')
        self.oo = par   #in mas

    def pass_position(self, ra, dec):
        self.ra = ra
        self.dec = dec

    def pass_magnitude(self, mag, err=None, band='Ks'):
        self.m = mag
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
                distance=(1000/self.oo)*units.pc)
        #Find the extinction coefficient
        Av = bayestar(coords, mode='median')

        return Av * coeffs[self.band]

    def get_M(self):
        if self.check_contents():
            print('Im going to kill the run so you can pass the values correctly.')
            sys.exit()
            return None
        m = self.m       #Call in the magnitude
        mu0 = self.get_mu0()  #Call in the distance modulus
        A = self.get_A()      #Call in the extinction coefficient

        return m - mu0 - A

if __name__ == '__main__':
    #Lets start with the class
    Cyg = Star('Cyg')
    Cyg.pass_parallax(1.0)#mas
    Cyg.pass_position(0.,0.)
    Cyg.pass_magnitude(10,band='Afasdf')
    M = Cyg.get_M()
