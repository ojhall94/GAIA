#!/usr/bin/python
#Oliver J. Hall 2017

import sys
import numpy as np
import matplotlib.pyplot as plt
import glob as glob
import pandas as pd

import barbershop


if __name__ == "__main__":

    sfile = glob.glob('../data/AM_TRI/k1.7*.txt')[0]
    dff = pd.read_csv(sfile,sep='\s+')
    dff['Aks'] = 0.114*dff.Av
    dff['M_ks'] = dff.Ksmag - dff.mu0 - dff.Aks

    dff = dff[dff.M_ks < -0.5]
    dff = dff[dff.M_ks > -2.5]
    dff = dff[dff.Ksmag < 15.]
    dff = dff[dff.Ksmag > 6.]

    print dff.info()

    barber=barbershop.open(dff, 'M_ks','Ksmag')
    barber.add_client('label',lower=4, upper=4)
    barber.add_client('Mass')
    barber.add_client('M_H')
    barber.add_client('logTe')
    barber.histograms_on(x=True)

    plt.scatter(dff.logTe, dff.logL, c=dff.Mass)

    barber.show_mirror()

    sys.exit()


    sfile = glob.glob('../data/Elsworth+/Elsworth*.csv')[0]
    dff = pd.read_csv(sfile)

    print(dff.info())


    barber = barbershop.open(dff,'astero_parallax','kic_kmag')
    barber.add_client('kic_radius')
    barber.add_client('numstage')
    plt.hist(dff.kic_radius,histtype='step',bins='sqrt')


    barber.show_mirror()

    sys.exit()


    plt.close('all')
    sfile = glob.glob('Repo_Data/*all*.txt')[0]
    odf = pd.read_csv(sfile, sep='\s+')

    '''Correct data'''
    odf['Aks'] = 0.114*odf.Av
    odf['M_ks'] = odf.Ks - odf['m-M0'] - odf.Aks
    odf['Aj'] = 0.282*odf.Av
    odf['M_j'] = odf.J - odf['m-M0'] - odf.Aj
    odf['JKs'] = odf.J - odf.Ks
    odf['rad'] = np.sqrt((10**odf.logL)/(10**odf.logTe)**4)
    #
    # '''Set first order cuts on data'''
    odf = odf[odf.M_ks < -0.5]
    odf = odf[odf.M_ks > -2.5]
    odf = odf[odf.Ks < 15.]
    odf = odf[odf.Ks > 6.]



    print(odf.info())

    barber = barbershop.open(odf[::10],'M_ks','Ks')
    barber.histograms_on(x=True,y=True)
    barber.add_client('Mact')
    barber.add_client('[M/H]')
    barber.add_client('M_ks')
    barber.add_client('stage',lower=4,upper=4)

    plt.hist(odf.rad,histtype='step',bins='sqrt')

    barber.show_mirror()

    sys.exit()

    sfile = glob.glob('../data/Elsworth+/gaia*.csv')[0]
    grd = pd.read_csv(sfile)
    sfile = glob.glob('../data/Elsworth+/Yvonne*.csv')[0]
    els = pd.read_csv(sfile)

    grd['stage'] = ''
    grd['numstage'] = np.nan

    for row, kic in enumerate(grd.KIC):
        try:
            loc = np.where(els.KIC == kic)[0][0]
            grd['stage'][row] = els.evol_overall[loc]

            if grd.stage[row] == 'RC':
                grd['numstage'][row] = 1
            elif grd.stage[row] == '2CL':
                grd['numstage'][row] = 2
            elif grd.stage[row] == 'RGB':
                grd['numstage'][row] = 3
            elif grd.stage[row] == 'U':
                grd['numstage'][row] = 0
            else:
                pass

        except IndexError:
            pass
    grd.to_csv('../data/Elsworth+/Elsworth_x_TGAS.csv')
