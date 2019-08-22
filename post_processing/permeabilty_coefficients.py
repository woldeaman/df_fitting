# -*- coding: utf-8 -*-
"""Analysis for permeabilty coefficients."""
import numpy as np
import fitting_scripts.FPModel as fp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import mpltex  # for acs style figures


#################
#  ENVIRONMENT  #
##########################################################################
##########################################################################


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
# %%

##########################################################################


###############
#  MAIN LOOP  #
##########################################################################
D_gels = np.linspace(5, 55, 100)
dFs = np.linspace(0.5, 1.85, 100)
D, F = np.meshgrid(D_gels, dFs)
P_step = permeabilty_stepfunction(D, F)
P_sig = np.array([[permeabilty_sigmoidfunction(f, d, d_t=20, D_sol=150) for d in D_gels] for f in dFs])
# make figure
comparison_plot(D, F, P_sig, P_step, save=True)
##########################################################################
