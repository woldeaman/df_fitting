# -*- coding: utf-8 -*-
"""Analyse D and F results of fits."""
import numpy as np
import pandas as pd
import fitting_scripts.FPModel as fp
import scipy.special as sp
import matplotlib.pyplot as plt
import matplotlib.animation as an
import scipy.optimize as op
import functools as ft
import mpltex  # for acs style figures
import sys


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
def compute_amount(discretizations, z_vectors, c_exps, scalings, d_sols, d_gels,
                   delta_fs, t_sig, d_sig, dextrans, t_max=50000, dt=None):
    """Compute averaged concentration in three different segments."""
    if dt is None:  # standart dt = 10 s
        dt = {g: {d: 10 for d in dextrans[g]} for g in gels}
    avg_gel_theo, avg_trans_theo, avg_bulk_theo = {}, {}, {}
    avg_gel_exp, avg_trans_exp, avg_bulk_exp = {}, {}, {}
    for g in gels:
        avg_gel_theo[g], avg_trans_theo[g], avg_bulk_theo[g] = {}, {}, {}
        avg_gel_exp[g], avg_trans_exp[g], avg_bulk_exp[g] = {}, {}, {}
        for dex, d_s, d_g, df, t_s, del_s in zip(dextrans[g], d_sols[g], d_gels[g],
                                                 delta_fs[g], t_sig[g], d_sig[g]):
            D = np.array([fp.sigmoidalDF([d_s[0], d_g[0]], t_s[0], del_s[0], z)
                          for z in z_vectors[g][dex]])
            F = np.array([fp.sigmoidalDF([0, df[0]], t_s[0], del_s[0], z)
                          for z in z_vectors[g][dex]])
            segments = np.concatenate((np.zeros(6), np.arange(D.size))).astype(int)
            D, F = fp.computeDF(D, F, shape=segments)
            # computing WMatrix, start smaller than 6, because D, F is const. only there
            W = fp.WMatrixVar(D, F, start=4, end=None, deltaXX=discretizations[g][dex], con=True)
            # gather interface position in bin dimensions
            z_trans = int(np.round(t_s[0]/10))  # dz = 10 and 6 bins in bulk
            # compute numerical profiles for longer time points
            tt_long = np.logspace(-2, np.log10(t_max), 5000).astype(int)
            cc_theo = [fp.calcC(c_exps[g][dex][0], t=t, W=W) for t in tt_long]
            # now compute average concentration in different segments
            bulk = [np.average(cc[:6]) for cc in cc_theo]  # regime without experimental data
            trans = [np.average(cc[6:(z_trans+6)]) for cc in cc_theo]  # experimental data regime up to interface
            gel = [np.average(cc[(z_trans+6):]) for cc in cc_theo]  # everything beyond interface
            # store computed amounts with corresponding time points
            avg_gel_theo[g][dex] = np.c_[tt_long, gel]
            avg_trans_theo[g][dex] = np.c_[tt_long, trans]
            avg_bulk_theo[g][dex] = np.c_[tt_long, bulk]
            # compute and store also experimental data
            tt_exp = np.arange(0, len(c_exps[g][dex])*dt[g][dex], dt[g][dex])
            trn_exp = [np.average(c_exps[g][dex][0][6:(z_trans+6)])]
            trn_exp += [np.average(f*cc[:z_trans]) for cc, f in zip(c_exps[g][dex][1:], scalings[g][dex])]
            gel_exp = [np.average(c_exps[g][dex][0][(z_trans+6):])]
            gel_exp += [np.average(f*cc[z_trans:]) for cc, f in zip(c_exps[g][dex][1:], scalings[g][dex])]
            # store computed amounts with corresponding time points
            avg_gel_exp[g][dex] = np.c_[tt_exp, gel_exp]
            avg_trans_exp[g][dex] = np.c_[tt_exp, trn_exp]

    return avg_bulk_theo, avg_trans_theo, avg_gel_theo, avg_gel_exp, avg_trans_exp


def discretizations_and_initial_profiles(path, dextrans):
    """Gather and re-compute discretizations and initial profiles for each setup."""
    discretizations, z_vectors, c_exp = {}, {}, {}
    for g in gels:  # cycle through all data and read xx-vectors
        discretizations[g], c_exp[g], z_vectors[g] = {}, {}, {}
        for dex in dextrans[g]:
            data = np.loadtxt(path+'/gel%i_dex%i/gel%i_dex%i.txt' % (g, dex, g, dex), delimiter=',')
            zz, cc = data[:, 0], data[:, 1:]  # extract data
            dxx_dist, dxx_width = fp.discretization_Block(zz)  # compute discretization
            cc_complete = fp.build_zero_profile(cc)  # build t=0 profile
            discretizations[g][dex] = dxx_dist  # storing discretizations
            c_exp[g][dex] = cc_complete  # storing all profiles
            z_vectors[g][dex] = zz  # storing z vectors

    return discretizations, c_exp, z_vectors


def read_results(path, dextrans):
    """Read fit results from analysed data."""
    D_sol, D_gel, dF, t_sig, d_sig = {}, {}, {}, {}, {}  # storing data in dicts
    scalings = {}
    for g in gels:
        d_s, d_g, f, t_s, del_s = [], [], [], [], []
        scalings[g] = {}
        for dex in dextrans[g]:  # cycle through all analyses
            data = pd.read_excel(path+'/gel%i_dex%i/results.xlsx' % (g, dex))
            # read and store from excel file
            parameters = np.array([data['Averaged Results'][:5].values,
                                   data['Standart Deviation'][:5].values])
            scales = np.loadtxt(path+'/gel%i_dex%i/scalings_avg.txt' % (g, dex), delimiter=',')[:, 2]
            for i, store in enumerate([d_s, d_g, f, t_s, del_s]):
                store.append(parameters[:, i])  # save to list first
            scalings[g][dex] = scales
        D_sol[g], D_gel[g], dF[g] = np.array(d_s), np.array(d_g), np.array(f)
        t_sig[g], d_sig[g] = np.array(t_s), np.array(del_s)

    return D_sol, D_gel, dF, t_sig, d_sig, scalings


def hydrodynamic_radius(M_w, x=0.33, a=0.46):
    """
    Compute hydrodynamic radius from molar mass.
    Based on the Mark-Houwink equation, compute the hydrodynamic radius as a
    function of the molar mass. Default values for paramters are for dextran,
    taken from Aimar et al., Journal of Membrane Science, 1990.

    x       -   pre-factor for conversion
    a       -   exponent for conversion
    M_w     -   molar mass [g/mol]
    R_h     -   hydrodynamic/Stokes radius [Å]
    """
    R_h = x * M_w**a  # based on Mark-Houwink equation and relation of R_h to gyradius

    return R_h


def free_energy_theory(r_mol, r_pore, mode='uniform'):
    """
    Compute the partition coefficient based on the molecular dimensions.
    Using the derived equation for spherical diffusing molecules and uniform
    or random pore network from Giddings et al., J Phys Chem, 1968.

    r_mol       -   radius of the diffusing molecule
    r_pore      -   (effective) radius of the pores
    mode        -   assuming 'uniform' or 'random' pores
    """
    if mode is 'uniform':
        df = -2*np.log(1 - (r_mol/r_pore))  # ratio of pore and diffusor length is significant
    elif mode is 'random':
        df = (r_mol**2)/(r_pore**2)  # ratio of pore and diffusor area is significant
    else:
        sys.exit('Error: Supplied mode not recognized!')
    return df


def diffusivity_change(r_mol, r_pore):
    """
    Compute the reduction in diffusivity between bulk and hydrogel.
    Based on equation from Renkin, J Gen Physio, 1954, compute ratio of diffusivities
    in gel and in bulk d_gel/d_bulk.

    r_mol       -   radius of the diffusing molecule
    r_pore      -   radius of the pores
    ratio       -   ratio = d_gel/d_bulk
    """
    s = r_mol/r_pore  # ratio of the two dimensions for use in equation
    p = [-4.19, 3.87, 0, -1.372, -0.94813, 0, 2.08877, 0, -2.1444, 1]
    ratio = np.polyval(p, s)  # evaluate polynomial with coefficients from model

    return ratio


def fit_theory(dF):
    """Compute fit of free energy model to experimental data."""
    # first compute hydrodynamic radius, in Å
    radii = {g: [hydrodynamic_radius(m*1000) for m in dextrans[g]] for g in gels}
    # create separate functions for models to fit
    uni = ft.partial(free_energy_theory, mode='uniform')  # uniform pore network
    rand = ft.partial(free_energy_theory, mode='random')  # random rod pore network
    r_pore, K_theo, d_ratio_theo = {}, {}, {}
    r_h_theo = np.linspace(1, 40)  # r-range vector to plot
    for model, label in zip([uni, rand], ['uniform', 'random']):  # try both models
        # fit (effective) pore radius, only for 6kDa dextran, as with other only 2 points...
        r_pore[label] = op.curve_fit(model, radii[6], dF[6][:, 0], bounds=(0, np.inf), p0=100)
        K_theo[label] = np.c_[r_h_theo, np.exp(-model(r_h_theo, r_pore[label][0], mode=label))]
        d_ratio_theo[label] = np.c_[r_h_theo, diffusivity_change(r_h_theo, r_pore[label][0])]
    # saving hydrodynamic radius, fitted pore radius, theoretical data for partition coefficient & d-ratio
    return radii, r_pore, K_theo, d_ratio_theo


@mpltex.acs_decorator  # making acs-style figures
def figure_c_init(discretization, c_init, save=False, savePath=None):
    """Make figure for explaining building of initial profile."""
    # z ticks and labels
    zz_lin = np.array([np.sum(discretization[1:i]) for i in range(1, discretization.size)])
    zz_scale = zz_lin - zz_lin[6]  # zero is at bin 6
    # for labeling the x-axis correctly, first 4 bins at different separation
    zz = np.concatenate(([0, 6, 12, 18], np.arange(c_init.size-4)+19))  # generate z-vector for entire system
    zlabels = [np.append(zz[:4], zz[6::5]).astype(int),
               np.append(zz_scale[:4], zz_scale[6::5]).astype(int)]

    fig = plt.figure()  # make figure now
    ext = plt.plot(zz[:6], c_init[:6], 'ko--', mfc='white')
    plt.plot(zz[6:7], c_init[6:7], 'k--')
    exp = plt.plot(zz[6:], c_init[6:], 'ok--')
    plt.xticks(zlabels[0], zlabels[1])
    plt.ylabel('Normalized concentration')
    plt.xlabel('z-distance [$\mu$m]')
    fig.tight_layout(pad=0.5, w_pad=0.55)
    plt.legend([ext[0], exp[0]], ['extended', 'experiment'], frameon=False)

    if save:
        plt.savefig(savePath+'/c_init.eps')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def figure_scalings(zz_exp, c_exp, tt, scalings, skip=5, save=False, savePath=None):
    """Make figure comparing original experimental profiles and scaled ones."""
    # create appropriate colormap using dummy plot
    z = [tt/60, tt/60, tt/60]  # amplitude dummy is time
    dummy_map = plt.imshow(z, cmap='jet')

    fig = plt.figure()  # create figure with correctly colored profiles
    colors = [dummy_map.cmap(x) for x in np.linspace(0, 1, len(c_exp))]
    c_og, c_sc = [], []
    for f, c, col in zip(scalings[::skip], c_exp[1::skip], colors[::skip]):
        c_sc.append(plt.plot(zz_exp, f*c, '-', c=col))
        c_og.append(plt.plot(zz_exp, c, ':', c=col))
    # labels and legends
    plt.xlabel('z-distance [$\mu$m]')
    plt.ylabel('Normalized concentration')
    fig.tight_layout(pad=0.5, w_pad=0.55)
    plt.legend([c_og[0][0], c_sc[0][0]], ['original', 'scaled'], frameon=False)
    fig.colorbar(dummy_map, label='Time [min]', pad=0.0125)

    if save:
        plt.savefig(savePath+'/scaling_comparison.eps')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def figure_explanation(save=False, savePath=None):
    """Make figure for explaining the model."""
    # make exemplary sigmoidal shape function
    xx = np.linspace(0, 1)
    sigmoid = np.array([0.5*(1+sp.erf((x-0.5)/(np.sqrt(2)*0.1))) for x in xx])
    fig, axes = plt.subplots(1, 2, sharex='all')
    fig.text(0.005, 0.93, 'A', fontsize='xx-large', weight='extra bold')  # add subplot label
    fig.text(0.46, 0.93, 'B', fontsize='xx-large', weight='extra bold')
    axes[1].plot(xx, sigmoid, 'b-')
    axes[1].plot(xx, 1-sigmoid, 'r-')
    axes[1].axvline(0.5, ls=':', c='k')
    axes[1].axvspan(-0.1, 0.5, color=[0.875, 0.875, 1], lw=0)  # bulk = blue
    axes[1].axvspan(0.5, 1.1, color=[0.9, 0.9, 0.9], lw=0)  # gel = grey
    axes[1].text(0.02, 1.03, '$D_{\\text{sol}}$', color='r')
    axes[1].text(0.9, 0.03, '$D_{\\text{gel}}$', color='r')
    axes[1].text(0.9, 1.03, '$F_{\\text{gel}}$', color='b')
    axes[1].text(0.02, 0.03, '$F_{\\text{sol}} = 0$', color='b')
    axes[1].text(-0.19, 0.48, 'Diffusivity', color='r', rotation=90)
    axes[1].text(-0.2, 0.5, '/', rotation=90)
    axes[1].text(-0.19, 0.99, 'Free Energy', color='b', rotation=90)
    axes[1].text(0.46, -0.065, '$2d_{\\text{int}}$')
    axes[1].annotate('', (0.4, 0.05), (0.6, 0.05), arrowprops=dict(arrowstyle='<->'))
    axes[1].set(xlabel='z-distance')
    axes[1].set_xlim([-0.1, 1.1])
    axes[1].set_ylim([-0.1, 1.15])
    axes[1].set_yticks([])
    axes[1].set_xticks([0, 0.1, 0.5, 1])
    axes[1].set_xticklabels(['$-z_{\\text{top}}$', 0, '$z_{\\text{int}}$', '$z_{\\text{bot}}$'])
    axes[0].remove()

    # for double column figures in acs style format
    w_double = 7  # inch size for width of double column figure for ACS journals
    width, height = fig.get_size_inches()
    fig.set_size_inches(w_double, height)
    fig.tight_layout(pad=0.5, w_pad=0.55)

    if save:
        plt.savefig(savePath+'/model_intro.eps')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def figure_results(gels, dextrans, D_sol, D_gel, dF, save=False, dscale='linear',
                   xscale='linear', savePath=None, locs_dLegend=['upper center', 'upper right'],
                   name='DF_results'):
    """Plot results in nice figure."""
    gel_styles = {6: '-o', 10: 's--'}  # plotting styles for different gels
    meas_col = ['m', 'c']  # colors for different measurements
    if xscale in 'log':  # share x-axis
        share_x = False
    else:
        share_x = True

    fig, axes = plt.subplots(1, 3, sharex=share_x)
    fig.text(0.005, 0.92, 'A', fontsize='xx-large', weight='extra bold')  # add subplot label
    fig.text(0.33, 0.92, 'B', fontsize='xx-large', weight='extra bold')
    fig.text(0.67, 0.92, 'C', fontsize='xx-large', weight='extra bold')

    # plot diffusivities first
    for ax, dat, ylab in zip(axes, [D_sol, D_gel, dF],
                             ['$D_{\\text{sol}}$ [$\mu$m$^2$/s]',
                              '$D_{\\text{gel}}$ [$\mu$m$^2$/s]',
                              '$\Delta F$ [$k_{\\text{B}}T$]']):
        for d, dex, col in zip(dat, dextrans, meas_col):
            for g, mfcs in zip(gels, [col, 'white']):
                ax.errorbar(dex[g], d[g][:, 0], d[g][:, 1],
                            fmt=col+gel_styles[g], mfc=mfcs)
                ax.set_xticks([4, 10, 20, 40, 60, 70])
                ax.set_ylabel(ylab)
                ax.set_xlabel('$M_{\\text{dex}}$ [kDa]')

    # setting diffusivity yscale
    for ax in axes[:-1]:
        ax.set_yscale(dscale)
        ax.set_xscale(xscale)

    # dummy plots for legend
    plts = [[plt.plot([None], '%s%s' % (gel_styles[g], col), mfc=g_mfc)
             for g, g_mfc in zip(gels, [col, 'white'])]
            for col in meas_col]
    leg1 = axes[0].legend([p[0] for p in plts[0]],
                          ['$M_{\\text{gel}}$ = %d kDa' % g for g in gels],
                          frameon=False, title='\\underline{Measurement 2}', loc=locs_dLegend[0],
                          fontsize='small', markerscale=0.75, handlelength=1.2)
    axes[0].legend([p[0] for p in plts[1]],
                   ['$M_{\\text{gel}}$ = %d kDa' % g for g in gels],
                   frameon=False, title='\\underline{Measurement 3}', loc=locs_dLegend[1],
                   fontsize='small', markerscale=0.75, handlelength=1.2)
    axes[0].add_artist(leg1)

    # for double column figures in acs style format
    w_double = 7  # inch size for width of double column figure for ACS journals
    width, height = fig.get_size_inches()
    fig.set_size_inches(w_double, height)
    fig.tight_layout(pad=0.5, w_pad=0.55)

    if save:
        plt.savefig(savePath+'/%s.eps' % name)
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def figure_theory(r_h, D_sol_exp, D_gel_exp, dF_exp, D_ratio_theo, K_theo,
                  save=False, savePath=None):
    """Plot comparison to fitted theory data."""
    # compute experimental data to be plotted first
    K_exp = {g: [np.exp(-dF_exp[g][:, 0]),  # partition coefficient
                 dF_exp[g][:, 1]*np.exp(-dF_exp[g][:, 0])] for g in gels}
    D_ratio_exp = {g: [D_gel_exp[g][:, 0]/D_sol_exp[g][:, 0],  # ratio of diffusivities
                       np.sqrt((D_gel_exp[g][:, 1]/D_sol_exp[g][:, 0])**2 +
                               (D_sol_exp[g][:, 1]*D_gel_exp[g][:, 0]/(D_sol_exp[g][:, 0])**2)**2)]
                   for g in gels}
    gel_styles = {6: 'o', 10: 's'}  # plotting styles for different gels
    theo_styles = {'uniform': '-', 'random': '--'}  # styles for theory plotting

    fig, axes = plt.subplots(1, 2, sharex='all')
    fig.text(0.005, 0.92, 'A', fontsize='xx-large', weight='extra bold')  # add subplot label
    fig.text(0.5, 0.92, 'B', fontsize='xx-large', weight='extra bold')
    # plot partition coefficient comparison first
    for g, mfcs in zip(gels, ['b', 'white']):  # experimental data
        axes[0].errorbar(r_h[g], K_exp[g][0], K_exp[g][1], fmt='b'+gel_styles[g], mfc=mfcs)
    for mode in K_theo.keys():
        axes[0].plot(K_theo[mode][:, 0], K_theo[mode][:, 1], 'b'+theo_styles[mode])
    # axes[0].set(xlabel='$r_{\\text{dex}}$ [Å]', ylabel='$K$')
    # dummy plots for legends
    theory, experiment = plt.plot([None], '-b'), plt.plot([None], 'ob')
    axes[0].legend([theory[0], experiment[0]], ['theory', 'experiment'])

    # now plot ratio of diffusivities
    for g, mfcs in zip(gels, ['r', 'white']):  # experimental data
        axes[1].errorbar(r_h[g], D_ratio_exp[g][0], D_ratio_exp[g][1], fmt='r'+gel_styles[g], mfc=mfcs)
    for mode in D_ratio_theo.keys():
        axes[1].plot(D_ratio_theo[mode][:, 0], D_ratio_theo[mode][:, 1], 'r'+theo_styles[mode])
    # axes[1].set(xlabel='$r_{\\text{dex}}$ [Å]', ylabel='$D_{\\text{gel}}/D_{\\text{sol}}$')
    axes[1].set_ylim([0, 1])
    # dummy plots for legends

    # for double column figures in acs style format
    w_double = 7  # inch size for width of double column figure for ACS journals
    width, height = fig.get_size_inches()
    fig.set_size_inches(w_double, height)
    fig.tight_layout(pad=0.5, w_pad=0.55)

    if save:
        plt.savefig(savePath+'/theory_comparison.eps')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def figure_amount_time(avg_bulk_theo, avg_trans_theo, avg_gel_theo, avg_gel_exp,
                       avg_trans_exp, dextrans, save=False, savePath=None):
    """Make figure for temporal evolution of average concentration in different segments."""
    for g in gels:
        for dex in dextrans[g]:
            fig = plt.figure()
            plt.title("$M_{\\text{gel}}$ = %i kDa   $M_{\\text{dex}}$ = %i kDa" % (g, dex))
            blk_t = plt.plot(avg_bulk_theo[g][dex][:, 0]/60, avg_bulk_theo[g][dex][:, 1], 'b-')
            trn_t = plt.plot(avg_trans_theo[g][dex][:, 0]/60, avg_trans_theo[g][dex][:, 1], 'r-')
            gel_t = plt.plot(avg_gel_theo[g][dex][:, 0]/60, avg_gel_theo[g][dex][:, 1], 'k-')
            plt.plot(avg_trans_exp[g][dex][:, 0]/60, avg_trans_exp[g][dex][:, 1], 'r.')
            gel_e = plt.plot(avg_gel_exp[g][dex][:, 0]/60, avg_gel_exp[g][dex][:, 1], 'k.')
            plt.xscale('log')
            # labels and legends
            plt.xlabel('$t$ [min]')
            plt.ylabel('$\\overline{c}$')
            fig.tight_layout(pad=0.5, w_pad=0.55)
            leg1 = plt.legend([blk_t[0], trn_t[0], gel_t[0]], ['far solution', 'near solution', 'gel'],
                              frameon=False, loc='center left')
            plt.legend([gel_e[0], gel_t[0]], ['experiment', 'theory'], frameon=False)
            plt.gca().add_artist(leg1)
            if save:
                plt.savefig(savePath+'/gel_%i_dex%i_penetration.eps' % (g, dex))
            else:
                plt.show()


@mpltex.acs_decorator  # making acs-style figures
def make_animation(dx_dist, zz_exp, c_init, Dsol, Dgel, dF, t_sig, d_sig,
                   t_max, name='video', savePath=None):
    """Make video of changing concentration profiles."""
    tt = np.arange(0, t_max)  # extended time vector
    # z ticks and labels
    zz_lin = np.array([np.sum(dx_dist[1:i]) for i in range(1, dx_dist.size)])
    zz_scale = zz_lin - zz_lin[6]  # zero is at bin 6
    # for labeling the x-axis correctly, first 4 bins at different separation
    zz = np.concatenate(([0, 6, 12, 18], np.arange(c_init.size-4)+19))  # generate z-vector for entire system
    zlabels = [np.append(zz[:4], zz[6::5]).astype(int),
               np.append(zz_scale[:4], zz_scale[6::5]).astype(int)]
    z_trans = np.round(t_sig/10 + 19 + 2)  # scale transition to new x-vector

    # compute propagator for system
    D = np.array([fp.sigmoidalDF([Dsol, Dgel], t_sig, d_sig, z) for z in zz_exp])
    F = np.array([fp.sigmoidalDF([0, dF], t_sig, d_sig, z) for z in zz_exp])
    segments = np.concatenate((np.zeros(6), np.arange(D.size))).astype(int)
    D, F = fp.computeDF(D, F, shape=segments)
    # computing WMatrix, start smaller than 6, because D, F is const. only there
    W = fp.WMatrixVar(D, F, start=4, end=None, deltaXX=dx_dist, con=True)

    # First set up the figure, the axis, and the plot element we want to animate
    fig = plt.figure()
    ax = plt.axes(xlim=(zz[0]-0.01*zz[-1], zz[-1]+0.01*zz[-1]), ylim=(-.05, 1.05))
    ax.axvspan(ax.get_xlim()[0], z_trans, color=[0.875, 0.875, 1], lw=0)  # bulk = blue
    ax.axvspan(z_trans, ax.get_xlim()[-1], color=[0.9, 0.9, 0.9], lw=0)  # gel = grey
    ax.axvline(z_trans, ls=':', c='k')  # indicate transition
    ax.set_xticks(zlabels[0])
    ax.set_xticklabels(zlabels[1])
    ax.set(ylabel='Normalized concentration', xlabel='z-distance [$\mu$m]')

    line, = ax.plot([], [], 'k--.')
    time_text = ax.text(0.65, 0.9, '', transform=ax.transAxes)

    # initialization function: plot the background of each frame
    def init():
        line.set_data([], [])
        time_text.set_text('')
        return line,

    # animation function.  This is called sequentially
    def animate(i):
        c_theo = fp.calcC(c_init, t=tt[i], W=W)
        line.set_data(zz, c_theo)
        time_text.set_text('Time = %.2f min' % (tt[i]/60))
        return line,

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = an.FuncAnimation(fig, animate, init_func=init,
                            frames=tt.size, interval=20, blit=True)
    fig.tight_layout(pad=0.5, w_pad=0.55)
    # save the animation as an mp4.  This requires ffmpeg or mencoder
    anim.save(savePath+'/%s_animation.mp4' % name, fps=60, extra_args=['-vcodec', 'libx264'])
##########################################################################


#################
#  ENVIRONMENT  #
##########################################################################
gels = [6, 10]  # molecular weight of the analyzed gels [kDa]
# previous batch
dextrans_1 = {6: [4, 10, 20], 10: [4, 10]}  # molecular weight of analyzed dextrans for the different gels
dextrans_2 = {6: [4, 20, 70], 10: [4, 20, 70]}  # molecular weight of analyzed dextrans for the different gels
dextrans_3 = {6: [4, 20, 40, 70], 10: [4, 20, 40, 70]}  # molecular weight of analyzed dextrans for the different gels
dt = {g: {4: 10, 20: 10, 40: 10, 70: 30} for g in gels}  # new time discretization
home = '/Users/woldeaman/'  # change home directory accordingly
path_to_data_1 = home+'/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/4.Batch/ComputedData/'
path_to_data_2 = home+'/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/6.Batch/ComputedData/'
path_to_data_3 = home+'/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/7.Batch/ComputedData/'
measurements = [path_to_data_1, path_to_data_2, path_to_data_3]  # gather paths for different measurements
dextrans_compt = [dextrans_1, dextrans_2, dextrans_3]
save_path = home+'/Desktop'  # by default save on Desktop
##########################################################################


###############
#  MAIN LOOP  #
##########################################################################
# read fit data
D_sol, D_gel, dF, t_sig, d_sig, scalings = [], [], [], [], [], []
for mes, dex in zip(measurements, dextrans_compt):  # gather data from different measurements
    d_sol, d_gel, df, t_s, d_s, scal = read_results(mes, dex)
    D_sol.append(d_sol)
    D_gel.append(d_gel)
    dF.append(df)
    t_sig.append(t_s)
    d_sig.append(d_s)
    scalings.append(scal)

# read discretizations for analysis
discretizations, c_exps, z_vectors = discretizations_and_initial_profiles(path_to_data_3, dextrans_3)

# compute time resolved average concentration
(avg_bulk_theo, avg_trans_theo,
 avg_gel_theo, avg_gel_exp, avg_trans_exp) = compute_amount(discretizations, z_vectors, c_exps, scalings[2],
                                                            D_sol[2], D_gel[2], dF[2], t_sig[2], d_sig[2], dextrans=dextrans_3, dt=dt)

# computing theoretical data
r_h, r_pore_fit, K_theo, d_ratio_theo = fit_theory(dF)

# plot data
figure_explanation(save=True, savePath=save_path)
# example = [10, 10]  # previous batch
example = [10, 20]
example_dt = 10
figure_c_init(discretizations[example[0]][example[1]], c_exps[example[0]][example[1]][0], save=True, savePath=save_path)
figure_scalings(z_vectors[example[0]][example[1]], c_exps[example[0]][example[1]],
                np.arange(0, len(c_exps[example[0]][example[1]])*example_dt, example_dt), scalings[example[0]][example[1]], save=True, savePath=save_path)
figure_results(gels, dextrans_compt, D_sol, D_gel, dF, save=True, savePath=save_path)
# log plot figure
figure_results(gels, dextrans_compt[1:], D_sol[1:], D_gel[1:], dF[1:], save=True, savePath=save_path, dscale='log', xscale='log',
               locs_dLegend=['upper right', 'lower left'], name='DF_results_log')
figure_amount_time(avg_bulk_theo, avg_trans_theo, avg_gel_theo, avg_gel_exp, avg_trans_exp, dextrans_3, save=True, savePath=save_path)
figure_theory(r_h, D_sol, D_gel, dF, d_ratio_theo, K_theo, save=True, savePath=save_path)


t_max = 12000  # max video time for dextrans in seconds
for g in gels:
    for i, dex in enumerate(dextrans_2[g]):
        make_animation(discretizations[g][dex], z_vectors[g][dex], c_exps[g][dex][0], D_sol[g][i, 0],
                       D_gel[g][i, 0], dF[g][i, 0], t_sig[g][i, 0], d_sig[g][i, 0], t_max,
                       name='gel%i_dex%i' % (g, dex), savePath=save_path)
