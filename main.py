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

import os
import sys
import pickle

__outdir__ = '/home/oliver/PhD/Gaia_Project/Output/  '

class run_stan:
    def __init__(self, _dat, _init=0., _runlabel='', _data='astero'):
        self.dat = _dat
        self.init = _init
        self.data = _data #Either astero or gaia
        self.runlabel = __outdir__+'_'+_runlabel+'_'+_data

    def build_metadata(self):
        if self.data == 'astero':
            self.pars = ['mu', 'sigma', 'Q', 'sigo']
            self.verbose = [r'$\mu_{RC}$',r'$\sigma_{RC}$',r'$Q$', r'$\sigma_o$']
            self.units = ['mag','mag','','mag']

        if self.data =='gaia':
            self.pars = ['mu', 'sigma', 'Q', 'sigo', 'L']
            self.verbose = [r'$\mu_{RC}$',r'$\sigma_{RC}$',r'$Q$', r'$\sigma_o$', r'$L$']
            self.units = ['mag','mag','','mag','pc']

    def read_stan(self):
        if _data = 'astero':
            model_path = 'asterostan.pkl'
            if os.path.isfile(model_path):
                self.sm = pickle.load(open(model_path, 'rb'))
            else:
                print('No stan model found')
                sys.exit()

        if _data = 'gaia':
            model_path = 'astrostan.pkl'
            if os.path.isfile(model_path):
                self.sm = pickle.load(open(model_path, 'rb'))
            else:
                print('No stan model found')
                sys.exit()

    def run_stan(self):
        if self.init != 0.:
            self.fit = self.sm.sampling(data = self.dat,
                        iter= 10000, chains=4,
                        init = [self.init, self.init, self.init, self.init])
        else:
            self.fit = self.sm.sampling(data = self.dat,
                        iter= 10000, chains=4)

    def out_corner(self):
        chain = np.array([fit[label] for label in self.pars])
        corner.corner(chain.T,labels=self.verbose,\
                      quantiles=[0.16, 0.5, 0.84],\
                      show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig(self.runlabel+'_corner.png')
        plt.close('all')

    def out_stanplot(self):
        self.fit.plot()
        plt.savefig(self.runlabel+'_stanplot.png')
        plt.close('all')

    def run_output(self):
        #Save the chains
        chain = np.array([fit[label] for label in self.pars])
        np.savetxt(self.runlabel+'_chains.txt')

        #Save the parameters
        pardict = {label:np.median(self.fit[label]) for label in self.pars}
        pardict = pd.DataFrame.from_dict(pardict,orient='index').T
        pardict.to_csv(self.runlabel+'_pars.csv')

        #Save the Rhat values
        s = fit.summary()
        print(s['summary'][:,-1])
        rhat = s['summary'][:,-1]
        np.savetxt(self.runlabel+'_rhats.txt')

        #Plot the Rhat distribution
        rhatfin = rhat[np.isfinite(rhat)]
        sns.distplot(rhatfin)
        plt.title('Distribution of Rhat values')
        plt.savefig(self.runlabel+'_rhatdist.png')
        plt.close('all')

    def __call__(self, verbose=True, visual=True):
        self.build_metadata()
        self.read_stan()
        self.run_stan()

        if visual:
            self.out_corner()
            self.out_stanplot()

        if verbose:
            self.run_output()

        print('Run to + '+self.runlabel+' complete!')

if __name__ == "__main__":
    print('hi')
    '''
    I am a hacker now
    '''

    hack(the past)
