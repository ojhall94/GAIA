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
import glob
import argparse
parser = argparse.ArgumentParser(description='Run our PyStan model on some data')
parser.add_argument('type', type=str, choices=['astero', 'gaia'], help='Choice of PyStan model.')
parser.add_argument('iters', type=int, help='Number of MCMC iterations in PyStan.')
parser.add_argument('corrections', type=str, choices=['None', 'RC'], help='Choice of corrections to the seismic scaling relations.')
parser.add_argument('band', type=str, choices=['K','J','H','GAIA'], help='Choice of photometric passband.')
parser.add_argument('tempdiff', type=float, help='Perturbation to the temperature values in K')
# parser.add_argument('bclabel', type=str, choices=['nn','lt','nt'], help='Temp arg: nn: no prop; lt prop logg and teff; nt prop t only.')
parser.add_argument('-t', '--testing', action='store_const', const=True, default=False, help='Turn on to output results to a test_build folder')
parser.add_argument('-u','--update', action='store_const', const=True, default=False, help='Turn on to update the PyStan model you choose to run')
parser.add_argument('-a','--apokasc', action='store_const', const=True, default=False, help='Turn on to run on the APOKASC subsample')
parser.add_argument('-af', '--apofull', action='store_const', const=True, default=False, help='Turn on to propagate full APOKASC data')
parser.add_argument('-v', '--visual', action='store_const', const=True, default=False, help='Turn on to include cornerplots')
args = parser.parse_args()

import os
import sys
sys.path.append(os.path.expanduser('~')+'/PhD/Hacks_and_Mocks/asfgrid/')
import asfgrid

from omnitool.literature_values import Av_coeffs, hawkvals
from omnitool import scalings
from omnitool.literature_values import Rsol


# __outdir__ = os.path.expanduser('~')+'/Projects/Oli/Output/'
# __datdir__ = os.path.expanduser('~')+'/Projects/Oli/Data/'

__outdir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/Output/'
__datdir__ = os.path.expanduser('~')+'/PhD/Gaia_Project/data/KepxDR2/'

__iter__ = args.iters

def create_astrostan(overwrite=True):
    astrostan = '''
    functions {
        real bailerjones_lpdf(real r, real L){
            return log((1/(2*L^3)) * (r*r) * exp(-r/L));
        }
        real precalc_multinormal_lpdf(vector oo, vector oo_true, real logdetc, matrix invc, int N, real Nfloat){
            vector[N] r;
            r = oo - oo_true;

            return -0.5 * ((r' * invc * r) + logdetc + Nfloat * log(2*pi()));
        }
    }
    data {
        int<lower = 0> N;
        real<lower= 0> Nfloat;
        vector[N] m;
        vector<lower=0>[N] m_err;
        vector[N] oo;
        vector<lower=0>[N] RlEbv;

        matrix[N, N] invc;
        real logdetc;

        real mu_init;
        real mu_spread;
        real sig_init;
        real sig_spread;

    }
    parameters {
        //Hyperparameters
        real mu;
        real<lower=0.> sigma;
        real<lower=1.> sigo;
        real<lower=0.5,upper=1.> Q;
        real<lower=.1, upper=4000.> L;
        real oo_zp;

        //Latent parameters
        vector[N] M_infd_std;
        vector[N] Ai;
        vector<lower = 1.>[N] r_infd;
    }
    transformed parameters{
        //Inferred and transformed parameters
        vector[N] M_infd;

        //Operations
        for (n in 1:N){
            M_infd[n] = mu + sigma * M_infd_std[n]; //Rescale the M fit
        }
    }
    model {
        //Define calculable properties
        vector[N] m_true;
        vector[N] oo_true;

        //Hyperparameters [p(theta_rc, L)]
        mu ~ normal(mu_init, mu_spread); // Prior from seismo
        sigma ~ normal(sig_init, sig_spread);
        Q ~ normal(1., .25);
        sigo ~ normal(3.0, 1.0);
        L ~ uniform(0.1, 4000.);   // Prior on the length scale
        oo_zp ~ normal(0.0, 500.); // Prior on the offset (in mu as)

        //Latent parameters [p(alpha_i | theta_rc, L)]
        Ai ~ normal(RlEbv, 0.05);
        for (n in 1:N){
            r_infd[n] ~ bailerjones(L);
            target += log_mix(Q,
                normal_lpdf(M_infd_std[n] | 0., 1.),
                normal_lpdf(M_infd_std[n] | 0., sigo));
        }

        //Calculable properties
        for (n in 1:N){
            m_true[n] = M_infd[n] + 5*log10(r_infd[n]) - 5 + Ai[n];
            oo_true[n] = (1000./r_infd[n]) + (oo_zp/1000.);
        }

        //Observables [p(D | theta_rc, L, alpha)]
        oo ~ precalc_multinormal(oo_true, logdetc, invc, N, Nfloat);
        m ~ normal(m_true, m_err); //Measurement uncertainty on magnitude
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
        vector[N] Mobs;
        vector[N] Munc;
        real muH;
    }
    parameters {
        //Hyperparameters
        real mu;
        real <lower=0.> sigma;
        real <lower=0.5,upper=1.> Q;
        real <lower=1.> sigo;

        //Latent Parameters
        vector[N] Mtrue_std;
    }
    transformed parameters{
        vector[N] Mtrue;

        for (n in 1:N){
            Mtrue[n] = mu + sigma * Mtrue_std[n];
        }
    }
    model {
        mu ~ normal(muH, 1.0);  //p(theta)
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
        pkl_file =  open(model_path, 'wb')
        pickle.dump(sm, pkl_file)
        pkl_file.close()

    if not os.path.isfile(model_path):
        print('Saving Stan Model')
        sm = pystan.StanModel(model_code = asterostan, model_name='astrostan')
        pkl_file =  open(model_path, 'wb')
        pickle.dump(sm, pkl_file)
        pkl_file.close()

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
                        iter= __iter__, chains=4, seed=24601,
                        init = [self.init, self.init, self.init, self.init])
        else:
            fit = sm.sampling(data = self.dat, seed=24601,
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

        #Save the full fit extract
        outlabel = self.runlabel+'_fullchain_dict.pkl'
        output = open(outlabel, 'wb')
        pickle.dump(fit.extract(), output)
        output.close()

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
    if args.type == 'gaia':
        sfile = __datdir__+'rcxyu18.csv'
    else:
        if args.apokasc:
            sfile = __datdir__+'rcxyuxapokasc2.csv'
        else:
            sfile = __datdir__+'rcxyu18.csv'
    df = pd.read_csv(sfile)
    return df

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

def read_astero_output(majorlabel, minorlabel, sort):
    loc = __outdir__+majorlabel+'/'+sort+'_'+str(float(minorlabel))+'_fullchain_dict.pkl'
    pkl_file = open(loc, 'rb')
    fit = pickle.load(pkl_file)
    pkl_file.close()

    M_infd = np.median(fit['Mtrue'],axis=0)
    M_infd_std = np.median(fit['Mtrue_std'], axis=0)
    return M_infd, M_infd_std

def get_basic_init(type='gaia'):
    '''Returns a basic series of initial guesses in PyStan format.'''
    init = {'mu':-1.7,
            'sigma':0.1,
            'Q':0.95,
            'sigo':4.}

    if type == 'gaia':
        init['L'] = 1000

    return init

def get_fdnu(df):
    asf = asfgrid.Seism()
    evstate = np.ones(len(df))*2
    logz = np.log10(df.Z.values)
    teff = df.Teff.values + args.tempdiff
    dnu = df.dnu.values
    numax = df.numax.values

    mass, radius = asf.get_mass_radius(evstate, logz, teff, dnu, numax)
    logg = asf.mr2logg(mass, radius)
    fdnu = asf._get_fdnu(evstate, logz, teff, mass, logg, fill_value='nearest')

    return fdnu

def kernel(ra, dec, sigma, p):
    '''
    p[0] : Offset
    p[1] : Exponential decay scale
    '''
    thetaij = np.sqrt(np.subtract.outer(ra, ra)**2 + np.subtract.outer(dec, dec)**2)
    cov = p[0] * np.exp(-thetaij / p[1])
    np.fill_diagonal(cov, np.diag(cov) + sigma**2)
    return cov

def get_covmatrix(df):
    p = [285*10**-6, 14.]
    Sigma = kernel(df.ra.values, df.dec.values, df.parallax_error.values, p)
    invc = np.linalg.inv(Sigma)
    logdetc = np.linalg.slogdet(Sigma)[1] * np.linalg.slogdet(Sigma)[0]

    return Sigma, invc, logdetc

def get_bcs(tempdiff):
    # if args.bclabel == 'nn':
    #     BCs = pd.read_csv(__datdir__+'BCs/casagrande_bcs_0.0_singular.csv')
    # elif args.bclabel == 'nt':
    #     BCs = pd.read_csv(__datdir__+'BCs/Logg_unperturbed/casagrande_bcs_'+str(tempdiff)+'.csv')
    # elif args.bclabel == 'lt':
    #     BCs = pd.read_csv(__datdir__+'BCs/Logg_perturbed/casagrande_bcs_'+str(tempdiff)+'.csv')
    if not args.apokasc:
        BCs = pd.read_csv(__datdir__+'BCs/Logg_perturbed/casagrande_bcs_'+str(tempdiff)+'.csv')
    elif args.apokasc:
        BCs = pd.read_csv(__datdir__+'BCs/APOKASC/casagrande_bcs_'+str(tempdiff)+'.csv')

    return BCs

if __name__ == "__main__":
    if args.update:
        update_stan(model=args.type)
        sys.exit()

    type = args.type
    corrections = args.corrections
    band = args.band
    tempdiff = args.tempdiff
    if corrections=='None':
        corr = '_noCorrection'
    elif corrections=='RC':
        corr = '_Clump'

    if not args.testing:
        if type == 'astero':
            df = read_data()
        if type == 'gaia':
            kdf = read_data()
            from sklearn.utils import shuffle
            df = shuffle(kdf, random_state=24601)[:50].reset_index()
            shuffle=True

    else:
        from sklearn.utils import shuffle
        df = shuffle(read_data(), random_state=24601)[:100].reset_index()

    if type == 'astero':
        #Use asfgrid to calculate the correction to the scaling relations
        if corrections == 'None':
            fdnu = np.ones(len(df))
        if corrections == 'RC':
            if not args.apofull:
                fdnu = get_fdnu(df)
            else:
                fdnu = df.A_fdnu.values

        #Use omnitool to calculate magnitude, using precalculated bolometric corrections
        if not args.apokasc:
            SC = scalings(df.numax, df.dnu, df.Teff + tempdiff,
                        _numax_err = df.numax_err, _dnu_err = df.dnu_err, _Teff_err = df.Teff_err)
        if args.apokasc:
            if not args.apofull:
                SC = scalings(df.numax, df.dnu, df.A_Teff + tempdiff,
                            _numax_err = df.numax_err, _dnu_err = df.dnu_err, _Teff_err = df.A_Teff_err)
            if args.apofull:
                SC = scalings(df.A_numax, df.A_dnu, df.A_Teff + tempdiff,
                            _numax_err = df.A_numax_err, _dnu_err = df.A_dnu_err, _Teff_err = df.A_Teff_err)

        SC.give_corrections(fdnu = fdnu)

        BCs = get_bcs(tempdiff)
        df = pd.merge(df, BCs, how='left', on='KICID')

        bcerr = 0.02 # We assume an error of 0.02mag on the bolometric correction (roughly 1 to 2 %)
        Mobs = SC.get_bolmag() - df['BC_'+band]
        Munc = np.sqrt(SC.get_bolmag_err()**2 + bcerr**2)

        #Set up the data
        dat = {'N':len(df), 'Mobs':Mobs, 'Munc':Munc, 'muH':hawkvals[band]}

        #Set up initial guesses
        init = {'mu':hawkvals[band], 'sigma':0.1, 'Q':0.95, 'sigo':4.}

        if not args.testing:
            #Run a stan model on this. Majorlabel = the type of run, Minorlabel contains the temperature scale difference
            if args.apokasc:
                if not args.apofull:
                    run = run_stan(dat, init,
                                    _majorlabel='APOKASC_'+band+'_tempscale'+corr, _minorlabel=str(tempdiff), _stantype='astero')
                if args.apofull:
                    run = run_stan(dat, init,
                                    _majorlabel='APOFULL_'+band+'_tempscale'+corr, _minorlabel=str(tempdiff), _stantype='astero')

            else:
                run = run_stan(dat, init,
                                _majorlabel=band+'_tempscale'+corr, _minorlabel=str(tempdiff), _stantype='astero')
        else:
            print('Testing model...')
            run = run_stan(dat, init,
                            _majorlabel='test_build', _minorlabel=str(tempdiff)+'_'+band+corr, _stantype='astero')

        #Verbose = True saves chains, rhats, and median results. Visual=True saves cornerplot
        run(verbose=True, visual=args.visual)


    if type == 'gaia':
        if band == 'K':
            rlebv = df.Aks.values
            mband = df.Kmag.values
            merr = df.e_Kmag.values
        elif band == 'GAIA':
            rlebv = df.Ebv.values * 2.740 #As per Casagrande & Vandenberg 2018b
            #Correct the Gaia G mags as per Casagrande & Vandenberg 2018b
            mband = np.ones(len(df)) * df.GAIAmag.values
            sel = (mband > 6.) & (mband < 16.5)
            mband[sel] = 0.0505 + 0.9966*mband[sel]
            merr = np.ones(len(mband)) * 10.e-3 #Setting precision to 10mmag by default

        if not args.apokasc:
            astres = read_paramdict(band+'_tempscale_Clump', str(tempdiff), 'astero')
            M_infd, M_infd_std = read_astero_output(band+'_tempscale_Clump', str(tempdiff), 'astero')
        elif args.apokasc:
            astres = read_paramdict('APOKASC_'+band+'_tempscale_Clump', str(tempdiff), 'astero')
            M_infd, M_infd_std = read_astero_output('APOKASC_'+band+'_tempscale_Clump', str(tempdiff), 'astero')

        #Make sure these values are shuffled in in the right order
        if shuffle:
            kdf['M_infd'] = M_infd
            kdf['M_infd_std'] = M_infd_std
            shuf_kdf = shuffle(kdf, random_state=24601)[:100].reset_index()
            M_infd = shuf_kdf['M_infd']
            M_infd_std = shuf_kdf['M_infd_std']

        Sigma, invc, logdetc = get_covmatrix(df)

        dat = {'N':len(df),
               'Nfloat':np.float(len(df)),
                'm': mband,
                'm_err': merr,
                'oo': df.parallax.values,
                'RlEbv': rlebv,
                'logdetc': logdetc,
                'invc': invc,
                'mu_init': astres['mu'].values[0],
                'mu_spread': astres['mu_std'].values[0],
                'sig_init': astres['sigma'].values[0],
                'sig_spread': astres['sigma_std'].values[0]}}


        init= {'mu': astres.mu.values[0],
                'sigma': astres.sigma.values[0],
                'Q': astres.Q.values[0],
                'sigo': astres.sigo.values[0],
                'L': 1000.,
                'oo_zp':-29.,
                'M_infd':M_infd,
                'M_infd_std':M_infd_std,
                'r_infd':df.r_est,
                'Ai':rlebv}

        if not args.testing:
            #Run a stan model on this. Majorlabel = the type of run, Minorlabel contains the temperature scale difference
            if args.apokasc:
                run = run_stan(dat, init,
                                _majorlabel='Gaia_APOKASC_'+band+'_tempscale'+corr, _minorlabel=str(tempdiff), _stantype='gaia')

            else:
                run = run_stan(dat, init,
                                _majorlabel='Gaia_'+band+'_tempscale'+corr, _minorlabel=str(tempdiff), _stantype='gaia')
        else:
            print('Testing model...')
            run = run_stan(dat, init,
                            _majorlabel='test_build', _minorlabel=str(tempdiff)+'_'+band+corr, _stantype='gaia')



        run(verbose=True, visual=args.visual)
