
#!/usr/bin/python 

import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pyfits
import scipy.signal as sig
import scipy.stats as stats
import GRDdata
from matplotlib.widgets import Slider, Button, RadioButtons

point = ''

def rg_model_order(delta_nu, freq_zero, d02, d01, period_spacing, epsilon_g, gsplit, rsplit, coupling):
    zero = freq_zero
    ptwo = freq_zero - d02
    pone = freq_zero + d01
    # Calculate the mixed mode frequencies ...
    nu = np.arange(zero, zero + delta_nu, 0.005)
    nu *= 1e-6
    lhs = np.pi * (nu - (pone * 1e-6)) / (delta_nu*1e-6)
    rhs = np.arctan(coupling * np.tan(np.pi/(period_spacing * nu) - epsilon_g))
    mixed1 = np.zeros(100)
    counter = 0
    for i in np.arange(0, nu.size-1):
        if lhs[i] - rhs[i] < 0 and lhs[i+1] - rhs[i+1] > 0:
            mixed1[counter] = nu[i]
            counter += 1
    mixed1 = mixed1[:counter]
    # add in the rotational splitting ...
    splits = splitting(mixed1, delta_nu, period_spacing, \
                           gsplit, rsplit)
#    fig = plt.figure()
#    ax1 = fig.add_subplot(111)
#    l = ax1.plot(nu, lhs - rhs)
    mixed1 *= 1e6
    return zero, pone, ptwo, mixed1, mixed1 - splits, mixed1 + splits

def splitting(mixed, delta_nu, period_spacing, gsplit, rsplit):
    alpha_0 = (delta_nu * 1e-6)* period_spacing
    chi = 2.0 * mixed / (delta_nu*1e-6) * \
        np.cos(np.pi / (period_spacing * mixed))
    eta = 1.0 / (1.0 + alpha_0 * chi**2)
    splits = gsplit * (eta * (1 - 2.0*rsplit) + 2.0*rsplit)
    return splits

def update(val):
    period_spacing = speriod.val
    d01 = sd01.val
    d02 = sd02.val
    freq_zero = szero.val
    epsilon_g = sepsg.val
    gsplit = sgsplit.val
    rsplit = srsplit.val
    delta_nu = sdnu.val
    coupling = scoup.val
    z, one, two, mixed, minus, plus = \
        rg_model_order(delta_nu, freq_zero, d02, d01, \
                           period_spacing, epsilon_g, \
                           gsplit, rsplit, coupling)
    z_uni = universal_zero(sdnu.val, nu_max)
    mixed_plot.set_xdata(mixed)
    minus_plot.set_xdata(minus)
    plus_plot.set_xdata(plus)
    one_plot.set_xdata([one,one])
    two_plot.set_xdata([two,two])
    zero_plot.set_xdata([z,z])
    ax1.set_xlim(z - 0.25 * delta_nu, z + 0.85 * delta_nu)
    yscale = np.max(p[(z - 0.2 * delta_nu)/bw:(z + 0.8 * delta_nu)/bw])
    ax1.set_ylim(0, yscale)
    mixed_plot.set_ydata(np.ones(mixed.size)*yscale/3.0)
    minus_plot.set_ydata(np.ones(mixed.size)*yscale/3.3)
    plus_plot.set_ydata(np.ones(mixed.size)*yscale/3.3)
    uniz_plot.set_xdata(z_uni)
    uniz_plot.set_ydata(np.ones(np.size(z_uni)) * yscale / 4.0)
    fig.canvas.draw_idle()

def plot_universal_zero(ax, dnu, nu_max, top):
    n = np.arange(0,30,1)
    eps_zm = 0.626 + 0.538 * np.log10(dnu)
    eps_zp = 0.642 + 0.554 * np.log10(dnu)
    alpha_zm = 0.007
    alpha_zp = 0.009
    tmp_m = n + (0.0/2.0) + eps_zm + (alpha_zm / 2.0 * (n - (nu_max/dnu))**2)
    tmp_m *= dnu
    tmp_p = n + (0.0/2.0) + eps_zp + (alpha_zp / 2.0 * (n - (nu_max/dnu))**2)
    tmp_p *= dnu
    for i in n:
        p = ax.axvspan(tmp_m[i], tmp_p[i], facecolor='r', alpha=0.2)
    return tmp_m

def universal_zero(dnu, nu_max):
    n = np.arange(0,30,1)
    eps_z = 0.634 + 0.546 * np.log10(dnu)
    alpha_z = 0.008
    tmp = n + (0.0/2.0) + eps_z + (alpha_z / 2.0 * (n - (nu_max/dnu))**2)
    tmp *= dnu
    return tmp

def output(event):
    z, one, two, mixed, minus, plus = \
        rg_model_order(sdnu.val, szero.val, sd02.val, sd01.val, \
                           speriod.val, sepsg.val, \
                           sgsplit.val, srsplit.val, scoup.val)
    print 'Dnu ', sdnu.val
    print 'l=0 ', z
    print 'l=1p    ', one
    print 'l=2 ', two
    print 'Period_spacing ', speriod.val
    print 'epsilon_g ', sepsg.val
    print 'coupling ', scoup.val
    print 'g_split ', sgsplit.val
    print 'R_split ', srsplit.val
    print 'l=1mixed ', mixed
    print 'l=1m=1mixed ', minus
    print 'l=1m=-1mixed ', plus
    print '-----------------------------------------'

if __name__=="__main__":
    f, p, bw = GRDdata.get_psd(sys.argv[1])
    kic, dnu, nu_max, nu_width = GRDdata.get_global_txt(sys.argv[2])
    giant_type = 'rgb'
    if len(sys.argv) > 3:
        giant_type = sys.argv[3]
    mul = 3.0
    fmin = nu_max - mul * nu_width
    fmax = nu_max + mul * nu_width
    delta_nu = dnu
    freq_zero = nu_max
    d02 = 0.1 * delta_nu
    d01 = 0.5 * delta_nu
    period_spacing = 80.0 # seconds
    clump = 0
    epsilon_g = 0.0
    coupling = 0.2
    gsplit = 0.4
    rsplit = 0.5
    z, one, two, mixed, minus, plus = \
        rg_model_order(delta_nu, freq_zero, d02, d01, \
                           period_spacing, epsilon_g, \
                           gsplit, rsplit, coupling)
    top = np.max(p[(freq_zero - 0.2 * delta_nu)/bw:\
                       (freq_zero + 0.8 * delta_nu)/bw])
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.49, top=0.97, left=0.05, right=0.95)
    ax1 = fig.add_subplot(111)
    data = ax1.plot(f, p, 'k-')
    smoothed = ax1.plot(f, GRDdata.smooth_power(p, 12), 'r-')
    zero_plot, = ax1.plot([z,z], [0,top], 'r-', label='$\ell = 0$', \
                             linewidth=2)
    two_plot, = ax1.plot([two,two], [0,top], 'g-', label='$\ell_{p} = 2$')
    one_plot, = ax1.plot(np.ones(2)*one, [0,top], 'm-', label='$\ell_{p} = 1$')
    mixed_plot, = ax1.plot(mixed, np.ones(mixed.size)*top / 3, 'bo', \
                              label='$\ell_{mm} = 1$')
    minus_plot, = ax1.plot(minus, np.ones(mixed.size)*top / 3.3, 'b>', \
                             label='$\ell_{mm} = 1$')
    plus_plot, = ax1.plot(plus, np.ones(mixed.size)*top / 3.3, 'b<', \
                              label='$\ell_{mm} = 1$')
    ax1.set_ylim(0, top)
    # Plot the universal red-giant oscillation pattern
    # For just l=0
    z_uni = universal_zero(dnu, nu_max)
    uniz_plot, = ax1.plot(z_uni, np.ones(np.size(z_uni)) * top / 4.0, 'rp')
    # Define slider for period spacing ...
    print sys.argv
    print giant_type
    if giant_type == 'rgb':
        period_low = 50.0
        period_high = 100.0
    if giant_type == 'odd':
        period_low = 10.0
        period_high = 50.0
    if giant_type == 'clump':
        period_low = 150.0
        period_high = 350.0
        period_spacing = 220.0
        print 'Clump star!'
    axperiod = plt.axes([0.1, 0.07, 0.8, 0.03])
    speriod = Slider(axperiod, '$\mathrm{Period \; Spacing}$', period_low, period_high, \
                         valinit=period_spacing)
    # Define slider for d01
    axd01 = plt.axes([0.1, 0.12, 0.8, 0.03])
    sd01 = Slider(axd01, '$\mathrm{\delta_{0,1}}$', 0.3*delta_nu, 0.7*delta_nu, \
                         valinit=d01)
    # define slider for d02
    axd02 = plt.axes([0.1, 0.17, 0.8, 0.03])
    sd02 = Slider(axd02, '$\mathrm{\delta_{0,2}}$', 0.0*delta_nu, 0.2*delta_nu, \
                         valinit=d02)
    # define slider for zero frequency ...
    axzero = plt.axes([0.1, 0.32, 0.8, 0.03])
    szero = Slider(axzero, '$\mathrm{Freq \; \ell = 0}$', \
                       fmin, fmax, \
                         valinit=freq_zero)
    # define slider for g mode splitting ...
    axgsplit = plt.axes([0.1, 0.27, 0.8, 0.03])
    sgsplit = Slider(axgsplit, '$\mathrm{g \; split}$', \
                       0.0, 1.0, \
                         valinit=0.4)
    # define slider for splitting slope ...
    axrsplit = plt.axes([0.1, 0.22, 0.8, 0.03])
    srsplit = Slider(axrsplit, '$\mathrm{R}$', \
                       -1.0, 2.0, \
                         valinit=0.0)
    # define slider for epsilon_g ...
    axepsg = plt.axes([0.1, 0.02, 0.8, 0.03])
    sepsg = Slider(axepsg, '$\mathrm{\epsilon_{g}}$', \
                       0.0, 3.0, \
                         valinit=0.0)
    # define slider for delta_nu ...
    axdnu = plt.axes([0.1, 0.37, 0.8, 0.03])
    sdnu = Slider(axdnu, 'Dnu', \
                       dnu*0.95, dnu*1.05, \
                         valinit=dnu)
    # define slider for coupling ...
    axcoup = plt.axes([0.1, 0.42, 0.8, 0.03])
    scoup = Slider(axcoup, 'Coupling', \
                       0.0, 0.4, \
                         valinit=0.2)
    # Add in an output button ...
    axoutput = plt.axes([0.88, 0.93, 0.05, 0.03])
    button = Button(axoutput, 'Output', hovercolor='0.8')
    ax1.set_xlim(z - 0.25 * delta_nu, z + 0.85 * delta_nu)
    speriod.on_changed(update)
    sd01.on_changed(update)
    sd02.on_changed(update)
    szero.on_changed(update)
    sepsg.on_changed(update)
    sgsplit.on_changed(update)
    srsplit.on_changed(update)
    sdnu.on_changed(update)
    scoup.on_changed(update)
    button.on_clicked(output)
    plt.show()
