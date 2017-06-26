# !/usr/bin/env python
# -*- coding: utf-8 -*-
# Oliver J. Hall

import numpy as np
import pandas as pd

import glob
import sys
import corner as corner
from tqdm import tqdm

import matplotlib
# import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

from statsmodels.robust.scale import mad as mad_calc
from sklearn import linear_model
import scipy.stats as stats
import scipy.misc as misc

if __name__ == '__main__':

    sig = 0.138
    M = -1.62

    x = np.linspace(-2.,-0.25,10000)

    F = (1/(np.sqrt(2*np.pi) * sig)) * np.exp(-(x - M)**2/(2*sig**2))
    Fmax = (1/(np.sqrt(2*np.pi) * sig))
    F/=Fmax

    plt.plot(x,F,linewidth=2)
    plt.plot(x,F/2)
    plt.plot(x,F/4)
    plt.plot(x,F/8)
    plt.plot(x,F/16)
    plt.show()


    xp = np.linspace(10,7,1000)
    l = 1.
    P = np.exp(-l) * l**xp / misc.factorial(xp)
    m = P.max()

    for l in range(0,10):
        P = np.exp(-l) * l**xp / misc.factorial(xp)
        P /= m
        plt.plot(xp,P)
    plt.show()
