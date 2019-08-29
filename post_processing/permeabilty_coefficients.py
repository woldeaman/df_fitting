# -*- coding: utf-8 -*-
"""Analysis for permeabilty coefficients."""
import numpy as np
import fitting_scripts.FPModel as fp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpltex  # for acs style figures


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
# %%
def permeabilty_stepfunction(D_gel, dF):
    """Compute permeabilty coefficient for step function."""
    norm_P = D_gel/np.exp(dF)  # equation for normalized permeabilty

    return norm_P


def permeabilty_sigmoidfunction(dF, D_gel, d_t=15, D_sol=100):
    """Compute permeabilty coefficient for sigmoid function."""
    zz = np.linspace(0, 100, 100)  # z-vector ranges from z_t to z_t*2
    # compute F and D profile, only left half side first
    F_z = np.array([fp.sigmoidalDF([0, dF], 0, d_t, z) for z in zz])
    D_z = np.array([fp.sigmoidalDF([D_sol, D_gel], 0, d_t, z) for z in zz])
    # now flip and append for complete profile
    F_z = np.append(F_z, F_z[::-1])
    D_z = np.append(D_z, D_z[::-1])
    L_muc = 100*2  # length for normalizing permeabilty
    # now compute integral for permeability
    intgrnd, dz = np.exp(F_z)/D_z, abs(zz[1]-zz[0])
    inv_P = intgrnd.sum()*dz  # compute integral for 1/P
    norm_P = L_muc/inv_P  # compute normalized permeabilty [µm^2/s]

    return norm_P


@mpltex.acs_decorator  # making acs-style figures
def comparison_plot(D, F, p_sig, p_step, savePath='/Users/woldeaman/Desktop/', save=False):
    """Comparison plot of permeability coefficients."""
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    # make contour plots
    for ax, p in zip(axes, [p_step, p_sig]):
        im = ax.contourf(D, F, p, 15, vmin=0, vmax=40)
    # adjust subplots and add axis for colormap

    divider = make_axes_locatable(plt.gca())
    cbar_ax = divider.append_axes("right", "5%", pad="5%")
    fig.colorbar(im, cax=cbar_ax, label="P$\\cdot L_\\text{muc}$ [$\\mu$m$^2$/s]")  # plot colorbar

    # labels
    axes[0].title.set_text("step profiles")
    axes[1].title.set_text("sigmoid profiles")
    axes[0].set_ylabel("$\\Delta F [k_\\text{B}T]$")
    for ax in axes:
        ax.set_xlabel("$D_\\text{gel}$ [$\\mu$m$^2$/s]")
        ax.minorticks_on()

    # for double column figures in acs style format
    w_double = 7  # inch size for width of double column figure for ACS journals
    width, height = fig.get_size_inches()
    fig.set_size_inches(w_double, height)
    fig.tight_layout(pad=1, w_pad=0.5, h_pad=0.5)

    if save:
        plt.savefig(savePath+'/permeability_coefficient.pdf')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def plot_measured_data(D, F, p_step, D_fit, F_fit, D_fit_err, F_fit_err,
                       savePath='/Users/woldeaman/Desktop/', save=False):
    """Comparison plot of permeability coefficients."""
    fig, ax = plt.subplots()
    # make contour plots
    im = ax.contourf(D, F, p_step, 15)
    # plot fitted data into same axis
    plts = []
    for gel, sym, mc in zip([6, 10], ['o', 's'], ['r', 'white']):
        plot = ax.errorbar(D_fit[gel], F_fit[gel], yerr=F_fit_err[gel], xerr=D_fit_err[gel], fmt=f'{sym}r-', mfc=mc)
        plts.append(plot)

    # adjust subplots and add axis for colormap
    divider = make_axes_locatable(plt.gca())
    cbar_ax = divider.append_axes("right", "5%", pad="5%")
    fig.colorbar(im, cax=cbar_ax, label="P$\\cdot L_\\text{muc}$ [$\\mu$m$^2$/s]")  # plot colorbar
    # labels
    ax.set_ylabel("$\\Delta F [k_\\text{B}T]$")
    ax.set_xlabel("$D_\\text{gel}$ [$\\mu$m$^2$/s]")
    ax.minorticks_on()
    # legend
    ax.legend([p[0] for p in plts], ['dPG-G6', 'dPG-G10'], frameon=False)

    # for double column figures in acs style format
    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height)
    fig.tight_layout(pad=1)

    if save:
        plt.savefig(savePath+'/permeability_coefficient.pdf')
    else:
        plt.show()
# %%
##########################################################################


###############
#  MAIN LOOP  #
##########################################################################
D_gels = np.linspace(5, 55, 100)
dFs = np.linspace(0.5, 1.95, 100)
D, F = np.meshgrid(D_gels, dFs)
P_step = permeabilty_stepfunction(D, F)
P_sig = np.array([[permeabilty_sigmoidfunction(f, d, d_t=20, D_sol=150) for d in D_gels] for f in dFs])
# make figure
comparison_plot(D, F, P_sig, P_step, save=True)
# data from measurements, order is dex4, dex10, dex20, dex40
d_gel = {6: [49.34, 31.22, 24.40, 16.82], 10: [47.30, 34.82, 15.75, 8.73]}
d_gel_err = {6: [0.78, 1.20, 1.55, 1.50], 10: [0.32, 8.39, 0.10, 0.36]}
df = {6: [0.68, 0.90, 1.32, 1.82], 10: [0.66, 0.87, 0.97, 1.55]}
df_err = {6: [0.08, 0.07, 0.07, 0.05], 10: [0.03, 0.02, 0.13, 0.16]}
# make figure for roland
plot_measured_data(D, F, P_step, d_gel, df, d_gel_err, df_err, savePath='/Users/woldeaman/Desktop/', save=True)
##########################################################################
