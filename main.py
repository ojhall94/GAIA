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

class asterostan:
    def __init__(self, _dat, _init=None, _runlabel='test'):
        self.dat = _dat
        self.init = _init
        self.runlabel = _runlabel

    def read_stan():
        model_path = 'asterostan.pkl'
        if os.path.isfile(model_path):
            self.sm = pickle.load(open(model_path, 'rb'))
        else:
            print('No stan model found')
            sys.exit()

    def run_stan():
        self.fit = self.sm.sampling(data = self.dat,\
                    iter= 10000, chains=4, init = [init, init, init, init])

    def corner():
        chain = np.array([fit['mu'],fit['sigma'],fit['Q'],fit['muo'],fit['sigo']])
        corner.corner(chain.T,labels=['mu','sigma','Q','muo','sigo'],\
                      quantiles=[0.16, 0.5, 0.84],\
                      show_titles=True, title_kwargs={"fontsize": 12})
        plt.savefig(__outdir__+'_'+self.runlabel)

    def __call__(self):

if __name__ == "__main__":
    print('hi')
    '''
    I am a hacker now
    '''

    hack(the past)
