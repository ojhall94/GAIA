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

from main import read_paramdict

from omnitool.literature_values import Av_coeffs
from omnitool import scalings
from omnitool.literature_values import Rsol

__outdir__ = os.path.expanduser('~')+'/Projects/Oli/Output/'
__datdir__ = os.path.expanduser('~')+'/Projects/Oli/Data/'

# __outdir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/Output/'
# __datdir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/data/KepxDR2/'

__iter__ = int(sys.argv[2])

def create_astrostan(overwrite=True):
    astrostan = '''
    functions {
        real bailerjones_lpdf(real r, real L){
            return log((1/(2*L^3)) * (r*r) * exp(-r/L));
        }
    }
    data {
        int<lower = 0> N;
        real m[N];
        real<lower=0> m_err[N];
        real oo[N];
        real oo_err[N];
        real<lower=0> RlEbv[N];

        real mu_init;
        real mu_spread;
    }
    parameters {
        //Hyperparameters
        real mu;
        real<lower=0.> sigma;
        real<lower=1.> sigo;
        real<lower=0.5,upper=1.> Q;
        real<lower=.1, upper=4000.> L;

        //Latent parameters
        real M_infd_std[N];
        real Ai[N];
        real<lower = 1.> r_infd[N];

        // Parallax offset
        real oo_zp;
    }
    transformed parameters{
        //Inferred and transformed parameters
        real M_infd[N];

        //Operations
        for (n in 1:N){
            M_infd[n] = mu + sigma * M_infd_std[n]; //Rescale the M fit
        }
    }
    model {
        //Define calculable properties
        real m_true[N];
        real oo_exp[N];

        //Hyperparameters in true space [p(theta_rc, L)]
        mu ~ normal(mu_init, mu_spread); // Prior from seismo
        sigma ~ normal(0., 1.); // Spread from seismo
        Q ~ normal(1., .25);
        sigo ~ normal(3.0, 1.0);
        L ~ uniform(0.1, 4000.);

        //Latent parameters [p(alpha_i | theta_rc, L)]
        Ai ~ normal(RlEbv, 0.03);
        for (n in 1:N){
            r_infd[n] ~ bailerjones(L);
            target += log_mix(Q,
                normal_lpdf(M_infd_std[n] | 0., 1.),
                normal_lpdf(M_infd_std[n] | 0., sigo));
        }

        //Calculable properties
        for (n in 1:N){
            m_true[n] = M_infd[n] + 5*log10(r_infd[n]) - 5 + Ai[n];
            oo_exp[n] = (1000./r_infd[n]) + (oo_zp/1000.);
        }

        //Observables [p(D | theta_rc, L, alpha)]
        oo ~ normal(oo_exp, oo_err); //Measurement uncertainty on parallax
        m ~ normal(m_true, m_err); //Measurement uncertainty on magnitude

        //
        oo_zp ~ normal(0.0, 500.); // Prior on the offset!

    }

    '''
    model_path = 'astrostan.pkl'
    if overwrite:
        print('Updating Stan model')
        sm = pystan.StanModel(model_code = astrostan, model_name='astrostan')
        with open(model_path, 'wb') as f:
            pickle.dump(sm, f)

    if not os.path.isfile(model_path):
        print('Saving Stan Model')
        sm = pystan.StanModel(model_code = astrostan, model_name='astrostan')
        with open(model_path, 'wb') as f:
            pickle.dump(sm, f)


def create_asterostan(overwrite=True):
    asterostan = '''
    data {
        int<lower = 0> N;
        real Mobs[N];
        real Munc[N];
    }
    parameters {
        //Hyperparameters
        real mu;
        real <lower=0.> sigma;
        real <lower=0.5,upper=1.> Q;
        real <lower=1.> sigo;

        //Latent Parameters
        real Mtrue_std[N];
    }
    transformed parameters{
        real Mtrue[N];

        for (n in 1:N){
            Mtrue[n] = mu + sigma * Mtrue_std[n];
        }
    }
    model {
        mu ~ normal(-1.6, 1.0);  //p(theta)
        sigma ~ normal(0.0, 1.0); //''
        sigo ~ normal(3.0, 2.0);  //''
        Q ~ normal(1., 0.1);    //''

        Mobs ~ normal(Mtrue, Munc); //p(D | theta, alpha)

        //p(alpha | theta)
        for (n in 1:N)
            target += log_mix(Q,
                        normal_lpdf(Mtrue_std[n] | 0., 1.),
                        normal_lpdf(Mtrue_std[n] | 0., sigo));
    }
    '''
    model_path = 'asterostan.pkl'
    if overwrite:
        print('Updating Stan model')
        sm = pystan.StanModel(model_code = asterostan, model_name='astrostan')
        with open(model_path, 'wb') as f:
            pickle.dump(sm, f)

    if not os.path.isfile(model_path):
        print('Saving Stan Model')
        sm = pystan.StanModel(model_code = asterostan, model_name='astrostan')
        with open(model_path, 'wb') as f:
            pickle.dump(sm, f)

def update_stan(model='gaia'):
    if model == 'gaia':
        create_astrostan(overwrite=True)
    if model == 'astero':
        create_asterostan(overwrite=True)
    if model == 'both':
        create_astrostan(overwrite=True)
        create_asterostan(overwrite=True)

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
            self.pars = ['mu', 'sigma', 'Q', 'sigo', 'L', 'oo_zp']
            self.verbose = [r'$\mu_{RC} (mag)$',r'$\sigma_{RC} (mag)$',r'$Q$', r'$\sigma_o (mag)$', r'$L (pc)$', r'$\varpi_{zp} (\mu as)$']

    def read_stan(self):
        '''Reads the existing stanmodels'''
        if self.data == 'astero':
            model_path = 'asterostan.pkl'
            if os.path.isfile(model_path):
                sm = pickle.load(open(model_path, 'rb'))
            else:
                print('No stan model found')
                create_asterostan(overwrite=True)
                sys.exit()

        if self.data == 'gaia':
            model_path = 'astrostan.pkl'
            if os.path.isfile(model_path):
                sm = pickle.load(open(model_path, 'rb'))
            else:
                print('No stan model found')
                create_astrostan(overwrite=True)
                sys.exit()

        return sm

    def run_stan(self):
        '''Runs PyStan'''
        sm = self.read_stan()

        if self.init != 0.:
            fit = sm.sampling(data = self.dat,
                        iter= __iter__, chains=2,
                        init = [self.init, self.init])
        else:
            fit = sm.sampling(data = self.dat,
                        iter= __iter__, chains=2)

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
    # update_stan(model='gaia')
    # sys.exit()
    type = sys.argv[1]
    corrections = sys.argv[3]
    band = sys.argv[4]
    tempdiff = np.float(sys.argv[5])
    if corrections=='None':
        corr = '_noCorrection'
    elif corrections=='RC':
        corr = '_Clump'

    df = read_data()[:2000] #Call in the Yu+18 data

    if type == 'astero':
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


    if type == 'gaia':
        '''NOTE: NEED TO GENERALISE FOR OTHER THAN K'''
        if band == 'K':
            rlebv = 'Aks'
        elif band == 'GAIA':
            rlebv = 'Ag'
        majorlabel = band+'_tempscale'+corr
        minorlabel = str(tempdiff)

        astres = read_paramdict(majorlabel, minorlabel, 'astero')

        dat = {'N':len(df),
                'm': df[band+'mag'].values,
                'm_err': df['e_'+band+'mag'].values,
                'oo': df.parallax.values,
                'oo_err': df.parallax_error.values,
                'RlEbv': df[rlebv].values,
                'mu_init': astres.mu.values[0],
                'mu_spread': astres.mu_std.values[0]}
                # 'sigma_init': astres.sigma.values[0],
                # 'sigma_spread': astres.sigma_std.values[0]}

        init= {'mu': astres.mu.values[0],
                'sigma': astres.sigma.values[0],
                'Q': astres.Q.values[0],
                'sigo': astres.sigo.values[0],
                'L': 1000.,
                'oo_zp':-30.}

        print(init)
        # #Run a stan model on this. Majorlabel = the type of run, Minorlabel contains the temperature scale difference
        run = run_stan(dat, _init= init,
                        _majorlabel=majorlabel, _minorlabel=str(tempdiff), _stantype='gaia')

        #Verbose = True saves chains, rhats, and median results. Visual=True saves cornerplot and pystan plot
        run(verbose=True, visual=True)
