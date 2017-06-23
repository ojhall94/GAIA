import emcee
import numpy as np
from tqdm import tqdm
import scipy.interpolate as interpolate
import scipy.integrate as integrate


class MCMC:
    '''Modified MCMC Class, with some additional features & functions:
        -Allows reading for KDEs as walker distributions
        -Has a convergence check that initiates reruns
        -Dumps the sampler after completion
        -Has functions to return the Mixture Model posteriors and Bayes factor
    '''

    def __init__(self, _select, _params, _like, _prior,_start_kdes=0,_ntemps=1,_niter=500):
        self.niter = _niter
        self.ntemps = _ntemps
        self.nwalkers = 200
        self.start_params = np.array(_params)
        self.like = _like
        self.prior = _prior
        self.ndims = len(self.start_params)
        self.select = _select
        self.max_conv = 5
        self.conv_accept = 0.02
        self.start_kdes = _start_kdes

    def dump(self):
        '''Can be called to dump the sampler
        '''
        self.sampler.pool.close()

    def run(self):
        '''Main body of the EMCEE run
        '''
        self.sampler = emcee.PTSampler(self.ntemps, self.nwalkers, \
                                  self.ndims, \
                                  self.like, self.prior, \
                                       threads = 2)
        # print(self.ndims)
        p0 = np.zeros([self.ntemps, self.nwalkers, self.ndims])
        if type(self.start_kdes) == int:
            for i in range(self.ntemps):
                for j in range(self.nwalkers):
                    p0[i,j,:] = self.start_params * \
                                (1.0 + np.random.randn(self.ndims) * 0.0001)

        else:
            for i in range(self.ntemps):
                for j in range(self.ndims):
                    xs, ys = self.start_kdes[j]
                    cdf = integrate.cumtrapz(ys, xs,initial=0)
                    inv_cdf = interpolate.interp1d(cdf, xs)
                    p0[i,:,j] = inv_cdf(np.random.rand(self.nwalkers))


        print('\nBurning in...')
        for p1, lnpp, lnlp in tqdm(self.sampler.sample(p0, iterations=self.niter)):
            pass

        self.sampler.reset()
        print('\nRunning again...')
        for i in range(self.max_conv):
            for pp, lnpp, lnlp in tqdm(self.sampler.sample(p1, iterations=self.niter)):
                pass
            med = np.median(self.sampler.chain[0,:,-self.niter:-1,:],axis=0)
            conv = np.std(med, axis=0) / np.median(med, axis=0)
            if np.all(conv < self.conv_accept):
                break
        samples = self.sampler.chain[0,:,:,:].reshape((-1, self.ndims))

        # np.savetxt('../Output/samplechain_'+self.select+'.txt', samples)
        return np.array(samples)

    def postprob(self, X):
        '''Returns the foreground posterior
        '''
        norm = 0.0
        fg_pp = np.zeros(len(X.ravel()))
        bg_pp = np.zeros(len(X.ravel()))

        lotemp = self.sampler.chain[0,:,:,:]

        for i in range(lotemp.shape[0]):
            for j in range(lotemp.shape[1]):
                fg = self.like.lnlike_fg(lotemp[i,j])
                bg = self.like.lnlike_bg(lotemp[i,j])
                fg_pp += np.exp(fg - np.logaddexp(fg, bg)).ravel()
                bg_pp += np.exp(bg - np.logaddexp(fg, bg)).ravel()
                norm += 1
        bg_pp /= norm
        fg_pp /= norm


        return bg_pp, fg_pp

    def log_bayes(self, X):
        '''Returns the log of Bayes factor
        '''
        fg_pp, bg_pp = self.postprob(X)

        '''WARNING IGNORED'''
        with np.errstate(invalid='ignore'):
            lnK = np.log(fg_pp) - np.log(bg_pp)

        return lnK
