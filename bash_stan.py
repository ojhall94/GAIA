#!/bin/env python
# -*- coding: utf-8 -*-
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

import pickle
import os
import sys
import glob

from omnitool.literature_values import Av_coeffs
from omnitool import scalings
from omnitool.literature_values import Rsol

__outdir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/Output/'
__datdir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/data/KepxDR2/'

__iter__ = int(sys.argv[1])

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
            # self.out_stanplot(fit)

        if verbose:
            self.run_output(fit)

        print('Run to + '+self.runlabel+' complete!')

def read_data():
    '''Reads in the Yu et al. 2018 data'''
    sfile = __datdir__+'rcxyu18.csv'
    df = pd.read_csv(sfile)
    return df

def get_basic_init(type='gaia'):
    '''Returns a basic series of initial guesses in PyStan format.'''
    init = {'mu':-1.7,
            'sigma':0.1,
            'Q':0.95,
            'sigo':4.}

    if type == 'gaia':
        init['L'] = 1000

    return init

if __name__ == "__main__":
    corrections = sys.argv[2]
    band = sys.argv[3]
    tempdiff = np.float(sys.argv[4])

    if corrections=='None':
        corr = '_noCorrection'
    elif corrections=='RC':
        corr = '_Clump'

    df = read_data()[:500] #Call in the Yu+18 data

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
    run(verbose=True, visual=True)
