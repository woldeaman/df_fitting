# -*- coding: utf-8 -*-
"""Analyse D and F results of fits."""
import numpy as np
import pandas as pd
import scipy.special as sp
import matplotlib.pyplot as plt
import mpltex  # for acs style figures


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
        plt.savefig(savePath+'DF_fit_results.eps')
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
    axes[1].text(0.46, -0.065, '2$\Delta z$')
    axes[1].annotate('', (0.4, 0.05), (0.6, 0.05), arrowprops=dict(arrowstyle='<->'))
    axes[1].set(xlabel='z-distance')
    axes[1].set_xlim([-0.1, 1.1])
    axes[1].set_ylim([-0.1, 1.15])
    axes[1].set_yticks([])
    axes[1].set_xticks([0, 0.5, 1])
    axes[1].set_xticklabels(['$z_{\\text{start}}$', '$z_{\\text{trans}}$', '$z_{\\text{end}}$'])
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
# plot data
figure_results(gels, dextrans, D_sol, D_gel, dF, save=True, savePath=home+'/Desktop/')
figure_explanation(save=True, savePath=home+'/Desktop/')
