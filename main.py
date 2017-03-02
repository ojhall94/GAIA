# !/usr/bin/env python3
# -*- coding: utf-8 -*-
# Oliver J. Hall

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import glob
import sys

import astropy.coordinates as coord
import gaia.tap as gt

if __name__ == '__main__':

    '''
    Kepler FOV is 105 square degrees, so sqrt(115)/2 in each direction from the centre.
    RA: 19h 22m 40s
    Dec: +44 30' 00''
    '''

    '''
    Need the absolute-apparent magnitude conversion
    Need to see whether this gaia data is representative
    '''

    ra = (19. + 22./60. + 40./3600.)*15.
    dec = 44. + 30./60.
    r = np.sqrt(105)/2

    sources = gt.cone_search(ra, dec, r, table="gaiadr1.tgas_source")

    print(sources)

    fig, ax = plt.subplots()
    ax.scatter(sources['ra'], sources['dec'],s=1,c="#000000")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$\delta$")
    plt.show()
