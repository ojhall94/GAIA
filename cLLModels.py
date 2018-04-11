
import numpy as np


class LLModels:
    '''Set of likelihood functions and models'''
    def __init__(self, _x, _y, _labels):
        self.x = _x
        self.y = _y

    def lorentzian(self, (x0, gamma), dim='x'):
        '''A simple lortenzian in x or y space'''
        if dim == 'y':
            #Calculating the likelihood in the Y direction
            lnL = 2*np.log(gamma) - np.log(np.pi*gamma) - np.log((self.y-x0)**2 + gamma**2)
        if dim == 'x':
            #Calculating the likelihood in the X direction
            lnL = 2*np.log(gamma) - np.log(np.pi*gamma) - np.log((self.x-x0)**2 + gamma**2)
        return lnL

    def gaussian(self, (mu, sig), dim='x'):
        '''A simple gaussian in x space'''
        if dim == 'y':
            #Calculating the likelihood in the Y direction
            lnL = -0.5 * (((mu - self.y) / sig)**2 + 2*np.log(sig) +np.log(2*np.pi))
        if dim == 'x':
            #Calculating the likelihood in the X direction
            lnL = -0.5 * (((mu - self.x) / sig)**2 + 2*np.log(sig) +np.log(2*np.pi))
        return lnL


####---ADDITIONAL MODELS FROM RGBb (not yet adapted to new system)

    def exp_x(self, p):
        '''A normalised rising exponential probability in x space'''
        lambd = p[self.loclambd]

        #Calculating the likelihood in the X direction
        A = lambd * (np.exp(lambd*self.x.max()) - np.exp(lambd*self.x.min()))**-1
        lnLx = np.log(A) + lambd*self.x
        return lnLx

    def gauss_line_y(self, p):
        '''A fit in y for a straight line with gaussian noise.'''
        sigma = p[self.locsigma]
        m = p[self.locm]
        c = p[self.locc]

        M = self.x * m + c
        lnLy = -0.5 * (((self.y - M) / sigma)**2 + 2*np.log(sigma) + np.log(2*np.pi))
        return lnLy

    def bivar_gaussian(self, p):
        '''A bivariate gaussian function for probability in two dimensions.'''
        sigx = p[self.locsigx]
        sigy = p[self.locsigy]
        mx = p[self.locmx]
        my = p[self.locmy]
        rho = p[self.locrho]

        lnLxy = -np.log(2*np.pi) - np.log(sigx*sigy) - 0.5*np.log(1-rho**2) -\
                (1/(2*(1-rho**2))) * (\
                (self.x - mx)**2/sigx**2 + (self.y - my)**2/sigy**2 -\
                (2*rho*(self.x - mx)*(self.y - my))/(sigx * sigy))

        return lnLxy

    def return_bivar_sologauss(self, p):
        '''Returns the single axis gaussians of the bivariate'''
        sigx = p[self.locsigx]
        sigy = p[self.locsigy]
        mx = p[self.locmx]
        my = p[self.locmy]
        rho = p[self.locrho]

        lnLx = -0.5 * (((mx - self.x) / sigx)**2 + 2*np.log(sigx) +np.log(2*np.pi))
        lnLy = -0.5 * (((my - self.y) / sigy)**2 + 2*np.log(sigy) +np.log(2*np.pi))

        return (np.exp(lnLx), np.exp(lnLy))
