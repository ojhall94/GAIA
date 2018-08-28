import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pystan
import os
import sys
import random
import pickle

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

sns.distplot(M_obs)
sns.distplot(M_unc)
sns.distplot(M_true)
plt.show()

#RUN THE DATA
dat = {'N': npts,
        'Mobs': M_obs,
        'Munc': M_unc}
init = {'mu' : rmu,
      'sigma': rsigma,
       'sigo': rsigo,
       'Q' : rQ}

model_path = 'asterostan.pkl'
if os.path.isfile(model_path):
    sm = pickle.load(open(model_path, 'rb'))

fit = sm.sampling(data = dat, iter= 10000, chains=4, init=[init,init,init,init])
fit.plot()
plt.show()





sys.exit()
npts = 10000
d = np.random.randn(npts)

model = '''
data{
    int N;
    real d[N];
}
parameters {
    real mu;
    real<lower=0> sigma;
}
model{
    d ~ normal(mu, sigma);
}
'''
sm = pystan.StanModel(model_code=model)

dat = {'N':len(d),'d':d}

fit = sm.sampling(data = dat, iter= 10000, chains=4)
fit.plot()
plt.show()
