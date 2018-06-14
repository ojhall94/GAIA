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

__outdir__ = '/home/oliver/PhD/Gaia_Project/Output/'
__datdir__ = '/home/oliver/PhD/Gaia_Project/data/KepxDR2/'

__iter__ = 1000

class run_stan:
    def __init__(self, _dat, _init=0., _runlabel='', _data='astero'):
        self.dat = _dat
        self.init = _init
        self.data = _data #Either astero or gaia
        self.runlabel = __outdir__+_runlabel+'_'+_data

    def build_metadata(self):
        if self.data == 'astero':
            self.pars = ['mu', 'sigma', 'Q', 'sigo']
            self.verbose = [r'$\mu_{RC} (mag)$',r'$\sigma_{RC} (mag)$',r'$Q$', r'$\sigma_o (mag)$']

        if self.data =='gaia':
            self.pars = ['mu', 'sigma', 'Q', 'sigo', 'L']
            self.verbose = [r'$\mu_{RC} (mag)$',r'$\sigma_{RC} (mag)$',r'$Q$', r'$\sigma_o (mag)$', r'$L (pc)$']

    def read_stan(self):
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
        pardict = pd.DataFrame.from_dict(pardict,orient='index').T
        pardict.to_csv(self.runlabel+'_pars.csv')

        #Save the Rhat values
        s = fit.summary()
        print(s['summary'][:,-1])
        rhat = s['summary'][:,-1]
        np.savetxt(self.runlabel+'_rhats.txt', rhat)

        #Plot the Rhat distribution
        rhatfin = rhat[np.isfinite(rhat)]
        sns.distplot(rhatfin)
        plt.title('Distribution of Rhat values')
        plt.savefig(self.runlabel+'_rhatdist.png')
        plt.close('all')

    def __call__(self, verbose=True, visual=True):
        self.build_metadata()
        fit = self.run_stan()

        if visual:
            self.out_corner(fit)
            self.out_stanplot(fit)

        if verbose:
            self.run_output(fit)

        print('Run to + '+self.runlabel+' complete!')

def read_data():
    sfile = __datdir__+'rcxyu18.csv'
    df = pd.read_csv(sfile)
    return df

def run_ast_test():
    npts = 500
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

    run = run_stan(data, _init=init, _runlabel='test', _data='astero')
    run(verbose=True, visual=True)

def test_magzeropoint():


if __name__ == "__main__":
    run_ast_test()
