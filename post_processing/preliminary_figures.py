# -*- coding: utf-8 -*-
"""Analyse D and F results of fits."""
import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt
import scipy.optimize as op
import functools as ft
import mpltex  # for acs style figures
import sys


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
def read_results(path):
    """Read fit results from analysed data."""
    D_sol, D_gel, dF = {}, {}, {}  # storing data in dicts
    for g in gels:
        d_s, d_g, f = [], [], []
        for dex in dextrans[g]:  # cycle through all analyses
            data = pd.read_excel(path+'/gel%i_dex%i/results.xlsx' % (g, dex))
            # read and store from excel file
            parameters = np.array([data['Averaged Results'][:3].values,
                                   data['Standart Deviation'][:3].values])
            for i, store in enumerate([d_s, d_g, f]):
                store.append(parameters[:, i])  # save to list first
        D_sol[g], D_gel[g], dF[g] = np.array(d_s), np.array(d_g), np.array(f)

    return D_sol, D_gel, dF


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
    axes[1].text(0.46, -0.065, '$2d_{\\text{trans}}$')
    axes[1].annotate('', (0.4, 0.05), (0.6, 0.05), arrowprops=dict(arrowstyle='<->'))
    axes[1].set(xlabel='z-distance')
    axes[1].set_xlim([-0.1, 1.1])
    axes[1].set_ylim([-0.1, 1.15])
    axes[1].set_yticks([])
    axes[1].set_xticks([0, 0.1, 0.5, 1])
    axes[1].set_xticklabels(['$-z_{\\text{start}}$', 0, '$z_{\\text{trans}}$', '$z_{\\text{end}}$'])
    axes[0].remove()

    # for double column figures in acs style format
    w_double = 7  # inch size for width of double column figure for ACS journals
    width, height = fig.get_size_inches()
    fig.set_size_inches(w_double, height)
    fig.tight_layout(pad=0.5, w_pad=0.55)

    if save:
        plt.savefig(savePath+'model_intro.eps')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def figure_results(gels, dextrans, D_sol, D_gel, dF, save=False, savePath=None):
    """Plot results in nice figure."""
    gel_styles = {6: '-o', 10: 's--'}  # plotting styles for different gels
    diff_cols = ['m', 'r']  # colors for D_sol, D_gel

    fig, axes = plt.subplots(1, 2, sharex='all')
    fig.text(0.005, 0.92, 'A', fontsize='xx-large', weight='extra bold')  # add subplot label
    fig.text(0.5, 0.92, 'B', fontsize='xx-large', weight='extra bold')
    # plot diffusivities first
    for dat, col in zip([D_sol, D_gel], diff_cols):
        for g, mfcs in zip(gels, [col, 'white']):
            axes[0].errorbar(dextrans[g], dat[g][:, 0], dat[g][:, 1],
                             fmt=col+gel_styles[g], mfc=mfcs)
    axes[0].set(xlabel='$M_{\\text{dex}}$ [kDa]', ylabel='$D$ [$\mu$m$^2$/s]')
    # dummy plots for legends
    dSol, dGel = plt.plot([None], '%so-' % diff_cols[0]), plt.plot([None], '%so-' % diff_cols[1])
    axes[0].legend([dSol[0], dGel[0]], ['$D_{\\text{sol}}$', '$D_{\\text{gel}}$'])

    # now plot free energies
    for g, mfcs in zip(gels, ['b', 'white']):
        axes[1].errorbar(dextrans[g], dF[g][:, 0], dF[g][:, 1],
                         fmt='b'+gel_styles[g], mfc=mfcs)
    axes[1].axhline(0, ls=":", c='k')
    axes[1].set(xlabel='$M_{\\text{dex}}$ [kDa]', ylabel='$\Delta F$ [$k_{\\text{B}}T$]')
    # dummy plots for legends
    gel6, gel10 = plt.plot([None], 'b%s' % gel_styles[6]), plt.plot([None], 'b%s' % gel_styles[10], mfc='white')
    axes[1].legend([gel6[0], gel10[0]], ['$M_{\\text{gel}} = 6$ kDa', '$M_{\\text{gel}} = 10$ kDa'])

    # for double column figures in acs style format
    w_double = 7  # inch size for width of double column figure for ACS journals
    width, height = fig.get_size_inches()
    fig.set_size_inches(w_double, height)
    fig.tight_layout(pad=0.5, w_pad=0.55)

    if save:
        plt.savefig(savePath+'DF_results.eps')
    else:
        plt.show()


# @mpltex.acs_decorator  # making acs-style figures
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
        plt.savefig(savePath+'theory_comparison.eps')
    else:
        plt.show()
##########################################################################


#################
#  ENVIRONMENT  #
##########################################################################
home = '/Users/woldeaman/'  # change home directory accordingly
gels = [6, 10]  # molecular weight of the analyzed gels [kDa]
dextrans = {6: [4, 10, 20], 10: [4, 10]}  # molecular weight of analyzed dextrans for the different gels
path_to_data = home+'/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/4.Batch/ComputedData/rescaling_live/free_DSol'
##########################################################################


###############
#  MAIN LOOP  #
##########################################################################
# read data
D_sol, D_gel, dF = read_results(path_to_data)
# computing theoretical data
r_h, r_pore_fit, K_theo, d_ratio_theo = fit_theory(dF)

# plot data
figure_explanation(save=True, savePath=home+'/Desktop/')
figure_results(gels, dextrans, D_sol, D_gel, dF, save=True, savePath=home+'/Desktop/')
figure_theory(r_h, D_sol, D_gel, dF, d_ratio_theo, K_theo, save=False, savePath=home+'/Desktop/')
