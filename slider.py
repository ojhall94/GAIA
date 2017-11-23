#!/usr/bin/python
#Oliver J. Hall 2017

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.mlab as mlab
from matplotlib.widgets import Slider, Button, RadioButtons
import glob as glob
import pandas as pd

class cUpdate:
    def __init__(self,_dff):
        self.dff = _dff

    def get_update(self,loggmin, loggmax, zmin, zmax):
        out = self.dff[dff.logg < loggmax]
        out = out[out.logg > loggmin]
        out = out[out['[M/H]'] < zmax]
        out = out[out['[M/H]'] > zmin]
        return (out.M_ks.values, out.Ks.values)

if __name__ == "__main__":
    plt.close('all')
    sfile = glob.glob('../data/TRILEGAL_sim/*all*.txt')[0]
    dff = pd.read_csv(sfile, sep='\s+')

    '''Correct data'''
    dff['Aks'] = 0.114*dff.Av
    dff['M_ks'] = dff.Ks - dff['m-M0'] - dff.Aks

    '''Set first order cuts on data'''
    dff = dff[dff.M_ks < -0.5]
    dff = dff[dff.M_ks > -2.5]
    dff = dff[dff.Ks < 15.]
    dff = dff[dff.Ks > 6.]
    dff = dff[0:8000]

    '''Load in first order data'''
    U = cUpdate(dff)

    '''Set second order cuts on data'''
    loggmaxvalinit = dff.logg.max()
    loggminvalinit = dff.logg.min()
    zmaxvalinit = 0.5
    zminvalinit = -0.5

    '''Define initial subsample of second order data'''
    init = dff[dff.logg < loggmaxvalinit]
    init = init[init.logg > loggminvalinit]
    init = init[init['[M/H]'] < zmaxvalinit]
    init = init[init['[M/H]'] > zminvalinit]

    '''Define plottable variables of first subsample'''
    x = init.M_ks
    y = init.Ks
    cg = init.logg
    cm = init['[M/H]']

    '''Initiate plots'''
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,sharey=True)
    l = ax1.scatter(x, y, cmap='cool', c=cg, s=3)
    m = ax2.scatter(x, y, cmap='Oranges', c=cm, s=3)
    fig.colorbar(l, ax=ax1, label=r'log_{10}(g)')
    fig.colorbar(m, ax=ax2, label='[M/H]')

    ax2.set_xlabel(r"$M_{Ks}$")
    ax1.set_ylabel(r"$m_{Ks}$")
    ax1.grid()
    ax2.grid()
    ax1.set_axisbelow(True)
    ax2.set_axisbelow(True)
    fig.suptitle(r"TRILEGAL simulated data at magnitudes near the Red Clump")

    '''Initiate sliders'''
    Sfig, Sax = plt.subplots()
    Sfig.subplots_adjust(bottom=0.45)
    hist, bins = np.histogram(x,bins=int(np.sqrt(len(x))))
    b = Sax.bar(bins[:-1],hist,color='k')
    # _,_,h = Sax.hist(x,histtype='step',color='k',bins=int(np.sqrt(len(x))))
    Sax.set_xlabel(r"$M_{Ks}$")
    Sax.set_ylabel('Counts')
    Sax.set_title(r"Histogram in Absolute Magnitude for TRILGAL sample")

    axcolor = 'white'
    axloggmax = Sfig.add_axes([0.15, 0.05, 0.60, 0.03], facecolor=axcolor)
    axloggmin = Sfig.add_axes([0.15, 0.1, 0.60, 0.03], facecolor=axcolor)
    axzmax = Sfig.add_axes([0.15, 0.15, 0.60, 0.03], facecolor=axcolor)
    axzmin = Sfig.add_axes([0.15, 0.20, 0.60, 0.03], facecolor=axcolor)

    sloggmax = Slider(axloggmax, 'Max log(g)', round(dff.logg.min()), round(dff.logg.max()),valinit = loggmaxvalinit)
    sloggmin = Slider(axloggmin, 'Min log(g)', round(dff.logg.min()), round(dff.logg.max()),valinit = loggminvalinit)
    szmax = Slider(axzmax, 'Max [M/H]', round(dff['[M/H]'].min()), round(dff['[M/H]'].max()),valinit = zmaxvalinit)
    szmin = Slider(axzmin, 'Min [M/H]', round(dff['[M/H]'].min()), round(dff['[M/H]'].max()),valinit = zminvalinit)

    '''Update function'''
    def update(val):
        loggmax = sloggmax.val
        loggmin = sloggmin.val
        zmax = szmax.val
        zmin = szmin.val

        uu = np.vstack(U.get_update(loggmin, loggmax, zmin, zmax))
        l.set_offsets(uu.T)
        m.set_offsets(uu.T)
        hist, bins = np.histogram(uu[0],bins=int(np.sqrt(len(uu[0]))))
        [bar.set_height(hist[i]) for i, bar in enumerate(b)]
        [bar.set_x(bins[i]) for i, bar in enumerate(b)]
        Sax.relim()
        Sax.autoscale_view()
        # Sax.cla()
        # Sax.hist(uu[0],histtype='step',color='k',bins=int(np.sqrt(len(uu[0]))))

        l.set_clim([loggmin,loggmax])
        m.set_clim([zmin,zmax])
        fig.canvas.draw_idle()
        Sfig.canvas.draw_idle()

    '''Calling updates on the second order data'''
    sloggmax.on_changed(update)
    sloggmin.on_changed(update)
    szmax.on_changed(update)
    szmin.on_changed(update)

    '''Initiating reset buttons'''
    resetax = plt.axes([0.83, 0.025, 0.15, 0.04])
    zresetax = plt.axes([0.83, 0.125, 0.15, 0.04])
    gresetax = plt.axes([0.83, 0.225, 0.15, 0.04])

    buttonz = Button(zresetax, 'Reset [M/H]', color=axcolor, hovercolor='0.975')
    buttong = Button(gresetax, 'Reset log(g)', color=axcolor, hovercolor='0.975')
    button  = Button(resetax, 'Reset all', color=axcolor, hovercolor='0.975')

    '''Reset functions'''
    def reset(event):
        sloggmax.reset()
        sloggmin.reset()
        szmax.reset()
        szmin.reset()
    def resetg(event):
        sloggmax.reset()
        sloggmin.reset()
    def resetz(event):
        szmax.reset()
        szmin.reset()
    buttonz.on_clicked(resetz)
    buttong.on_clicked(resetg)
    button.on_clicked(reset)

    plt.show()
