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
import datetime
import os


def save(event):
    loggmax = sloggmax.val
    loggmin = sloggmin.val
    zmax = szmax.val
    zmin = szmin.val
    cmax = scolmax.val
    cmin = scolmin.val
    out = U.get_update(loggmin, loggmax, zmin, zmax, cmin, cmax, save=True)

    '''Setting up extra-repo output space'''
    if os.path.isdir('../Cuts_Data') == False:
        os.mkdir('../Cuts_Data')

    '''Saving data with cuts and date of file in filename'''
    date = datetime.datetime.now()
    out.to_csv('../Cuts_Data/cuts_MH_JKs_logg.txt', sep='\t')

    cuts = pd.DataFrame(columns=['Logg_Min','Logg_Max','[M/H]_Min','[M/H]_Max',\
                                'J-Ks_Min','J-Ks_Max'])
    cuts.loc[0] = [loggmin, loggmax, zmin, zmax, cmin, cmax]
    cuts.to_csv('../Cuts_Data/cuts_'+str(date.year)+str(date.month)+str(date.day)+'.txt', sep='\t')

    Lfig.savefig('../Cuts_Data/cut_losses_'+str(date.year)+str(date.month)+str(date.day)+'.png')

    plt.close('all')

class cUpdate:
    def __init__(self,_dff, _odf):
        self.dff = _dff
        self.odf = _odf

    def get_update(self,loggmin, loggmax, zmin, zmax, cmin, cmax, save=False):
        out = self.dff[dff.logg < loggmax]

        if save:
            out = self.odf[odf.logg < loggmax]

        out = out[out.logg > loggmin]
        out = out[out['[M/H]'] < zmax]
        out = out[out['[M/H]'] > zmin]
        out = out[out.JKs < cmax]
        out = out[out.JKs > cmin]

        if save:
            return out
        rc_new = len(out[out.stage==4])
        rgb_new = len(out[out.stage==3])
        alt_new = len(out[(out.stage != 3) & (out.stage != 4)])

        return np.vstack((out.M_ks.values, out.Ks.values)), (rc_new, rgb_new, alt_new)

'''Update function'''
def update(val):
    loggmax = sloggmax.val
    loggmin = sloggmin.val
    zmax = szmax.val
    zmin = szmin.val
    cmax = scolmax.val
    cmin = scolmin.val

    uu, ss = U.get_update(loggmin, loggmax, zmin, zmax, cmin, cmax)
    '''Update Plots'''
    l.set_offsets(uu.T)
    m.set_offsets(uu.T)
    c.set_offsets(uu.T)
    l.set_clim([loggmin,loggmax])
    m.set_clim([zmin,zmax])
    c.set_clim([cmin,cmax])

    '''Update Histogram'''
    Sax.clear()
    Sax.hist(init.M_ks,histtype='step',color='r',bins=int(np.sqrt(len(init.M_ks))),label='Histogram before cuts')
    Sax.legend(loc='best',fancybox=True)
    Sax.set_xlabel(r"$M_{Ks}$")
    Sax.set_ylabel('Counts')
    Sax.set_title(r"Histogram in Absolute Magnitude for TRILGAL sample")
    Sax.hist(uu[0],histtype='step',color='k',bins=int(np.sqrt(len(uu[0]))),label='Histogram after cuts')

    '''Update List'''
    RC_n.set_text('Total after cut: '+str(ss[0]))
    RC_p.set_text('Remaining: '+str(ss[0]*100/rc_total)+r"\%")
    RGB_n.set_text('Total after cut: '+str(ss[1]))
    RGB_p.set_text('Remaining: '+str(ss[1]*100/rgb_total)+r"\%")
    ALT_n.set_text('Total after cut: '+str(ss[2]))
    ALT_p.set_text('Remaining: '+str(ss[2]*100/alt_total)+r"\%")


    '''Draw Plots'''
    Lfig.canvas.draw_idle()
    Gfig.canvas.draw_idle()
    Mfig.canvas.draw_idle()
    Cfig.canvas.draw_idle()
    Sfig.canvas.draw_idle()

'''TODO:
-Add cuts to text plot
'''

if __name__ == "__main__":
    plt.close('all')
    sfile = glob.glob('Repo_Data/*all*.txt')[0]
    odf = pd.read_csv(sfile, sep='\s+')

    '''Correct data'''
    odf['Aks'] = 0.114*odf.Av
    odf['M_ks'] = odf.Ks - odf['m-M0'] - odf.Aks
    odf['JKs'] = odf.J - odf.Ks

    '''Set first order cuts on data'''
    odf = odf[odf.M_ks < -0.5]
    odf = odf[odf.M_ks > -2.5]
    odf = odf[odf.Ks < 15.]
    odf = odf[odf.Ks > 6.]

    # dff = odf[0:8000]
    dff = odf[:]
    
    '''Load in first order data'''
    U = cUpdate(dff, odf)

    '''Set second order cuts on data'''
    loggmaxvalinit = dff.logg.max()
    loggminvalinit = dff.logg.min()
    zmaxvalinit = 0.5
    zminvalinit = -0.5
    colmaxvalinit = dff.JKs.max()
    colminvalinit = dff.JKs.min()

    '''Define initial subsample of second order data'''
    init = dff[dff.logg < loggmaxvalinit]
    init = init[init.logg > loggminvalinit]
    init = init[init['[M/H]'] < zmaxvalinit]
    init = init[init['[M/H]'] > zminvalinit]
    init = init[init.JKs < colmaxvalinit]
    init = init[init.JKs > colminvalinit]

    '''Setting up stage contribution percentages'''
    rc_total = len(dff[dff.stage == 4])
    rgb_total = len(dff[dff.stage ==3])
    alt_total = len(dff[(dff.stage != 3) & (dff.stage != 4)])
    rc_init = len(init[init.stage==4])
    rgb_init = len(init[init.stage==3])
    alt_init = len(init[(init.stage != 3) & (init.stage != 4)])

    rc_perc = rc_init*100/rc_total
    rgb_perc = rgb_init*100/rgb_total
    alt_perc = alt_init*100/alt_total

    Lfig, Lax = plt.subplots(figsize=(2,4.5))
    Lax.get_xaxis().set_visible(False)
    Lax.get_yaxis().set_visible(False)

    rc_txt = 'Remaining: '+str(rc_perc)+r"\%"
    rgb_txt = 'Remaining: '+str(rgb_perc)+r"\%"
    alt_txt = 'Remaining: '+str(alt_perc)+r"\%"

    Lax.text(0.1, 0.9, b'Red Clump Stars:', color='red')
    Lax.text(0.1, 0.85, 'Initial total: '+str(rc_total))
    RC_n = Lax.text(0.1, 0.80, 'Total after cut: '+str(rc_init))
    RC_p = Lax.text(0.1,0.75,rc_txt)

    Lax.text(0.1, 0.60, b'RGB Stars:',color='red')
    Lax.text(0.1, 0.55, 'Initial total: '+str(rgb_total))
    RGB_n = Lax.text(0.1, 0.50, 'Total after cut: '+str(rgb_init))
    RGB_p = Lax.text(0.1,0.45,rgb_txt)

    Lax.text(0.1, 0.30, b'Other types:', color='red')
    Lax.text(0.1, 0.25, 'Initial total: '+str(alt_total))
    ALT_n = Lax.text(0.1, 0.20, 'Total after cut: '+str(alt_init))
    ALT_p = Lax.text(0.1, 0.15, alt_txt)

    Lax.text(0.1, 0.05, 'Total Stars: '+str(len(dff)))

    '''Define plottable variables of first subsample'''
    x = init.M_ks
    y = init.Ks
    cg = init.logg
    cm = init['[M/H]']
    cc = init.JKs

    '''Initiate plots'''
    Gfig, Gax = plt.subplots()
    Mfig, Max = plt.subplots()
    Cfig, Cax = plt.subplots()
    l = Gax.scatter(x, y, cmap='cool', c=cg, s=3)
    m = Max.scatter(x, y, cmap='winter', c=cm, s=3)
    c = Cax.scatter(x, y, cmap='plasma', c=cc, s=3)
    Gfig.colorbar(l, label=r"$log_{10}$(g)")
    Mfig.colorbar(m, label='[M/H]')
    Cfig.colorbar(c, label=r"J-$K_s$")

    Gax.set_xlabel(r"$M_{Ks}$")
    Max.set_xlabel(r"$M_{Ks}$")
    Cax.set_xlabel(r"$M_{Ks}$")
    Gax.set_ylabel(r"$m_{Ks}$")
    Max.set_ylabel(r"$m_{Ks}$")
    Cax.set_ylabel(r"$m_{Ks}$")

    Gax.grid()
    Max.grid()
    Cax.grid()
    Gax.set_axisbelow(True)
    Max.set_axisbelow(True)
    Cax.set_axisbelow(True)
    Gfig.suptitle(r"TRILEGAL simulated data at magnitudes near the Red Clump (log10(g))")
    Mfig.suptitle(r"TRILEGAL simulated data at magnitudes near the Red Clump ([M/H])")
    Cfig.suptitle(r"TRILEGAL simulated data at magnitudes near the Red Clump (J-Ks)")

    '''Initiate sliders'''
    Sfig, Sax = plt.subplots()
    Sfig.subplots_adjust(bottom=0.45)
    _,_,h = Sax.hist(x,histtype='step',color='k',bins=int(np.sqrt(len(x))))
    Sax.set_xlabel(r"$M_{Ks}$")
    Sax.set_ylabel('Counts')
    Sax.set_title(r"Histogram in Absolute Magnitude for TRILGAL sample")

    axcolor = 'white'
    axloggmax = Sfig.add_axes([0.15, 0.1, 0.60, 0.03], facecolor=axcolor)
    axloggmin = Sfig.add_axes([0.15, 0.14, 0.60, 0.03], facecolor=axcolor)
    axzmax = Sfig.add_axes([0.15, 0.18, 0.60, 0.03], facecolor=axcolor)
    axzmin = Sfig.add_axes([0.15, 0.22, 0.60, 0.03], facecolor=axcolor)
    axcolmax = Sfig.add_axes([0.15, 0.26, 0.60, 0.03], facecolor=axcolor)
    axcolmin = Sfig.add_axes([0.15, 0.30, 0.60, 0.03], facecolor=axcolor)

    sloggmax = Slider(axloggmax, 'Max log(g)', round(dff.logg.min()), round(dff.logg.max()),valinit = loggmaxvalinit)
    sloggmin = Slider(axloggmin, 'Min log(g)', round(dff.logg.min()), round(dff.logg.max()),valinit = loggminvalinit)
    szmax = Slider(axzmax, 'Max [M/H]', round(dff['[M/H]'].min()), round(dff['[M/H]'].max()),valinit = zmaxvalinit)
    szmin = Slider(axzmin, 'Min [M/H]', round(dff['[M/H]'].min()), round(dff['[M/H]'].max()),valinit = zminvalinit)
    scolmax = Slider(axcolmax, 'Max J-Ks', round(dff['JKs'].min()), round(dff['JKs'].max()),valinit = colmaxvalinit)
    scolmin = Slider(axcolmin, 'Min J-Ks', round(dff['JKs'].min()), round(dff['JKs'].max()),valinit = colminvalinit)

    '''Calling updates on the second order data'''
    sloggmax.on_changed(update)
    sloggmin.on_changed(update)
    szmax.on_changed(update)
    szmin.on_changed(update)
    scolmax.on_changed(update)
    scolmin.on_changed(update)

    '''Initiating reset buttons'''
    resetax = plt.axes([0.83, 0.16, 0.15, 0.04])
    zresetax = plt.axes([0.83, 0.21, 0.15, 0.04])
    gresetax = plt.axes([0.83, 0.26, 0.15, 0.04])
    cresetax = plt.axes([0.83, 0.31, 0.15, 0.04])

    saveax = plt.axes([0.15, 0.04, 0.83, 0.05])
    closeax = plt.axes([0.83, 0.10, 0.15, 0.04])

    buttonz = Button(zresetax, 'Reset [M/H]', color=axcolor, hovercolor='0.7')
    buttong = Button(gresetax, 'Reset log(g)', color=axcolor, hovercolor='0.7')
    buttonc = Button(cresetax, 'Reset J-Ks', color=axcolor, hovercolor='0.7')
    button  = Button(resetax, 'Reset all', color=axcolor, hovercolor='0.7')

    buttonsave = Button(saveax, 'Save Cut Data (with list of cuts)',color=axcolor, hovercolor='.7')
    buttonclose = Button(closeax, 'Close Plots', color=axcolor, hovercolor='red')
    '''Reset functions'''
    def reset(event):
        sloggmax.reset()
        sloggmin.reset()
        szmax.reset()
        szmin.reset()
        scolmax.reset()
        scolmin.reset()
    def resetg(event):
        sloggmax.reset()
        sloggmin.reset()
    def resetz(event):
        szmax.reset()
        szmin.reset()
    def resetc(event):
        scolmax.reset()
        scolmin.reset()
    def close(event):
        plt.close('all')

    buttonz.on_clicked(resetz)
    buttong.on_clicked(resetg)
    buttonc.on_clicked(resetc)
    button.on_clicked(reset)
    buttonsave.on_clicked(save)
    buttonclose.on_clicked(close)

    plt.show(block=False)
