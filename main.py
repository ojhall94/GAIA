#!/bin/env python
# -*- coding: utf-8 -*-
"""
Main body of code for the Hall et al. 2018 work

.. codeauthor:: Oliver James Hall <ojh251@student.bham.ac.uk>

Dependencies on personal code:
    omnitool <github.com/ojhall94/omnitool>
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
sns.set_palette('colorblind')
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)
matplotlib.rc('axes',labelsize=15)

import pandas as pd
import pystan
import corner
import random
import pickle
import os
import sys
import glob

from omnitool.literature_values import Av_coeffs
from omnitool import scalings
from omnitool.literature_values import Rsol, Lsol, stefboltz


__outdir__ = os.path.expanduser('~')+'/Projects/Oli/Output/'
__datdir__ = os.path.expanduser('~')+'/Projects/Oli/Data/KepxDR2/'

# __outdir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/Output/'
# __datdir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/data/KepxDR2/'

__iter__ = 10000

'''_____'''
def get_basic_init(type='gaia'):
    '''Returns a basic series of initial guesses in PyStan format.'''
    init = {'mu':-1.7,
            'sigma':0.1,
            'Q':0.95,
            'sigo':4.}

    if type == 'gaia':
        init['L'] = 1000

    return init

def read_paramdict(majorlabel, minorlabel='', sort='astero'):
    '''Reads in results for either:
        -A full run series (majorlabel) where the minorlabel is included as a
            column in the output.
        -A single run (majorlabel and minorlabel).

        Returns a pandas dataframe.
    '''
    loc = __outdir__+majorlabel+'/'

    if minorlabel != '':
        globlist = glob.glob(loc+sort+'_'+str(float(minorlabel))+'_*pars*.csv')
    else:
        globlist = glob.glob(loc+sort+'*_*pars*.csv')

    minorlabels = [os.path.basename(globloc).split('_')[1] for globloc in globlist]

    df = pd.DataFrame()
    for n, globloc in enumerate(globlist):
        sdf = pd.read_csv(globloc, index_col = 0)
        if minorlabels[n] != 'pars.csv':
            sdf[majorlabel] = minorlabels[n]
        df = df.append(sdf)

    return df.sort_values(by=majorlabel)

def read_data():
    '''Reads in the Yu et al. 2018 data'''
    sfile = __datdir__+'rcxyu18.csv'
    df = pd.read_csv(sfile)
    return df
'''_____'''

class run_stan:
    def __init__(self, _dat, _init=0., _majorlabel='', _minorlabel='', _stantype='astero'):
        '''Core PyStan class.
        Input __init__:
        _dat (dict): Dictionary of the data in pystan format.

        _init (dict): Dictionary of initial guesses in pystan format.

        _majorlabel (str): Name of the run set. This will be the name of the local
        directory the results are stored in.

        _minorlabel (str): Name of the individual run (i.e. a numeric value). This
        will be included in the title of all output files.

        _stantype (str): Stanmodel to be used, either 'astero' or 'gaia'.

        Input __call__:
        verbose (bool): If True: saves chains, median and standard deviations on
        parameter posteriors, and the rhat values (as well as plot of rhats)

        visual (bool): If True: saves cornerplot and the pystan chain plot.
        '''
        self.dat = _dat
        self.init = _init
        self.data = _stantype #Either astero or gaia
        self.runlabel = __outdir__+_majorlabel+'/'+_stantype+'_'+_minorlabel

        #Check folder exists, if not, overwrite
        if not os.path.exists(__outdir__+_majorlabel):
            os.makedirs(__outdir__+_majorlabel)

    def build_metadata(self):
        '''Builds label metadata for the run'''
        if self.data == 'astero':
            self.pars = ['mu', 'sigma', 'Q', 'sigo']
            self.verbose = [r'$\mu_{RC} (mag)$',r'$\sigma_{RC} (mag)$',r'$Q$', r'$\sigma_o (mag)$']

        if self.data =='gaia':
            self.pars = ['mu', 'sigma', 'Q', 'sigo', 'L']
            self.verbose = [r'$\mu_{RC} (mag)$',r'$\sigma_{RC} (mag)$',r'$Q$', r'$\sigma_o (mag)$', r'$L (pc)$']

    def read_stan(self):
        '''Reads the existing stanmodels'''
        if self.data == 'astero':
            model_path = 'asterostan.pkl'
            if os.path.isfile(model_path):
                sm = pickle.load(open(model_path, 'rb'))
            else:
                print('No stan model found')
                sys.exit()

        if self.data == 'gaia':
            model_path = 'astrostan.pkl'
            if os.path.isfile(model_path):
                sm = pickle.load(open(model_path, 'rb'))
            else:
                print('No stan model found')
                sys.exit()

        return sm

    def run_stan(self):
        '''Runs PyStan'''
        sm = self.read_stan()

        if self.init != 0.:
            fit = sm.sampling(data = self.dat,
                        iter= __iter__, chains=4,
                        init = [self.init, self.init, self.init, self.init])
        else:
            fit = sm.sampling(data = self.dat,
                        iter= __iter__, chains=4)

        return fit

    def out_corner(self, fit):
        chain = np.array([fit[label] for label in self.pars])
        corner.corner(chain.T,labels=self.verbose,\
                      quantiles=[0.16, 0.5, 0.84],\
                      show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig(self.runlabel+'_corner.png')
        plt.close('all')

    def out_stanplot(self, fit):
        fit.plot()
        plt.savefig(self.runlabel+'_stanplot.png')
        plt.close('all')

    def run_output(self, fit):
        #Save the chains
        chain = np.array([fit[label] for label in self.pars])
        np.savetxt(self.runlabel+'_chains.txt',chain)

        #Save the parameters
        pardict = {label:np.median(fit[label]) for label in self.pars}
        pardict.update({label+'_std':np.std(fit[label]) for label in self.pars})
        pardict = pd.DataFrame.from_dict(pardict,orient='index').T
        pardict.to_csv(self.runlabel+'_pars.csv')

        #Save the Rhat values
        s = fit.summary()
        rhat = s['summary'][:,-1]
        np.savetxt(self.runlabel+'_rhats.txt', rhat)


    def __call__(self, verbose=True, visual=True):
        self.build_metadata()
        fit = self.run_stan()

        if visual:
            self.out_corner(fit)
            self.out_stanplot(fit)

        if verbose:
            self.run_output(fit)

        print('Run to + '+self.runlabel+' complete!')


def test_ast():
    '''A test using synthetic data of the asteroseismic PyStan model.'''
    npts = 5000
    rQ = .60     #Mixture weighting
    rmu = -1.7   #Inlier mean
    rsigma = .05 #Inlier spread
    rmuo = rmu   #Outlier mean [Not a parameter in the model]
    rsigo = .35  #Outlier spread

    #Create a series of fractional errors that are similar to those in our data
    rf1 = np.random.randn(npts/2)*0.016 + 0.083   #First component is a Gaussian
    rf2 = np.random.exponential(.04, npts/2)+.05  #Second component is an exponential
    rf_unshuf = np.append(rf1, rf2)
    rf = np.array(random.sample(rf_unshuf,npts)) #Shuffle the values before drawing from them

    #Drawing the fractional uncertainties for the inlier and outlier sets
    fi = rf[:int(npts*rQ)]
    fo = rf[int(npts*rQ):int(npts*rQ) + int((1-rQ)*npts)]

    iM_true = np.random.randn(int(npts*rQ)) * rsigma + rmu
    iunc = np.abs(fi * iM_true)
    iM_obs = iM_true + np.random.randn(int(npts*rQ))*iunc
    oM_true = np.random.randn(int((1-rQ)*npts)) * rsigo + rmuo
    ounc = np.abs(fo * oM_true)
    oM_obs = oM_true + np.random.randn(int((1-rQ)*npts))*ounc

    M_obs = np.append(oM_obs, iM_obs)  #Observed data
    M_unc = np.append(ounc, iunc)      #Uncertainty on the above
    M_true = np.append(oM_true, iM_true)  #The underlying ruth

    #RUN THE DATA
    data = {'N': npts,
            'Mobs': M_obs,
            'Munc': M_unc}
    init = {'mu' : rmu,
          'sigma': rsigma,
           'sigo': rsigo,
           'Q' : rQ}

    run = run_stan(data, _init=init, _majorlabel='test', _stantype='astero')
    run(verbose=True, visual=True)

def run_gaia(df, init, majorlabel='gaia_test', band='Ks'):
    '''A basic run of the gaia PyStan model'''

    #Define the data
    maglabel = band+'mag'
    if band=='Ks':
        maglabel = 'Kmag'
    dat = {'N' : len(df),
            'm' : df[maglabel].values,
            'm_err' : df['e_'+maglabel].values,
            'oo_uncorr' : df.parallax.values,
            'oo_err' : df.parallax_error.values,
            'RlEbv' : df.Ebv.values * Av_coeffs[band].values,
            'oo_zp' : -0.029}

    run = run_stan(dat, _init=init,
                _majorlabel=majorlabel, _stantype='gaia')
    run(verbose=True, visual=True)

def test_magzeropoint(df, init, band = 'Ks'):
    '''Runs the Gaia Pystan model for various values of the parallax zeropoint.'''
    #Define the data
    maglabel = band+'mag'
    if band=='Ks':
        maglabel = 'Kmag'
    dat = {'N' : len(df),
            'm' : df[maglabel].values,
            'm_err' : df['e_'+maglabel].values,
            'oo_uncorr' : df.parallax.values,
            'oo_err' : df.parallax_error.values,
            'RlEbv' : df.Ebv.values * Av_coeffs[band].values,
            'oo_zp' : 0.}

    #Our aim here is to vary oo_zp to see how it changes the position of the Clump
    oo_zps = np.linspace(0.000, 0.050, 10)
    for n, oo_zp in enumerate(oo_zps):
        dat['oo_zp'] = oo_zp
        run = run_stan(dat, _init=init,
                        _majorlabel='oo_zp', _minorlabel=str(round(oo_zp, 4)), _stantype='gaia')
        run(verbose=True, visual=True)
        print('Completed run on oo_zp: '+str(oo_zp))
    print('Completed full run on '+str(len(oo_zps))+' different parallax zero points.')

def test_temperaturescales(corrections='None', band='K'):
    if corrections=='None':
        corr = '_noCorrection'
    elif corrections=='RC':
        corr = '_Clump'

    '''Runs the Gaia Pystan model for various values of the temperature scale.
    We'll do the Gaia G band in this instance.'''
    df = read_data()[:1000] #Call in the Yu+18 data
    tempdiffs = np.arange(-50, 60, 10)
    for n, tempdiff in enumerate(tempdiffs):
        #Use omnitool to calculate G-band magnitude magnitude, using a given radius
        SC = scalings(df, df.numax, df.dnu, df.Teff + tempdiff,
                        _numax_err = df.numax_err, _dnu_err = df.dnu_err, _Teff_err = df.Teff_err)
        SC.give_corrections(Rcorr = df['R'+corr]*Rsol, Rcorr_err = df['R'+corr+'_err']*Rsol)
        Mobs = SC.get_bolmag() - df['BC_'+band]
        Munc = np.sqrt(SC.get_bolmag_err()**2 + 0.02**2) #We assume an error of 0.02 on the bolometric correction

        dat = {'N':len(df), 'Mobs':Mobs, 'Munc': Munc}

        #Run a stan model on this. Majorlabel = the type of run, Minorlabel contains the temperature scale difference
        run = run_stan(dat, _init=get_basic_init(type='astero'),
                        _majorlabel=band+'_tempscale'+corr, _minorlabel=str(tempdiff), _stantype='astero')

        #Verbose = True saves chains, rhats, and median results. Visual=True saves cornerplot and pystan plot
        run(verbose=True, visual=False)

if __name__ == "__main__":
    knoc = read_paramdict('K_tempscale_noCorrection')
    krc = read_paramdict('K_tempscale_Clump')
    knoc['tempscale'] = knoc['K_tempscale_noCorrection'].str.strip()
    knoc['tempscale'] = knoc.tempscale.astype(float)
    krc['tempscale'] = krc['K_tempscale_Clump'].str.strip()
    krc['tempscale'] = krc.tempscale.astype(float)

    ky = np.linspace(knoc.mu.max(), knoc.mu.min())
    kx = np.linspace(knoc.tempscale.min(), knoc.tempscale.max())

    plt.errorbar(knoc.tempscale, knoc.mu, yerr = knoc.mu_std, fmt='o', capsize=5, label='No Correction')
    plt.errorbar(krc.tempscale, krc.mu, yerr = krc.mu_std, fmt='o',  capsize=5,label='RC Correction')
    plt.plot(kx, ky)
    plt.xlabel('Change in Temperature Scale (K)')
    plt.ylabel('Position of RC in K band')
    plt.legend(fontsize=20)
    plt.title('Position of the RC in absolute K-band mag for a change in Temperature scale.', fontsize=15)
    plt.show()
