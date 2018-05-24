"""Analyse fits with different constant diffusivity."""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mpltex  # for acs style figures

# change home directory accordingly
home = '/Users/woldeaman/'


# script estimates experimental error and compares it to error for different
# diffusivity values in bulk
#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
@mpltex.acs_decorator  # making acs-style figures
def figure_diffusivities(d_sol, d_gel, f_gel, error, name='params_dsol',
                         title='', save=False, scale='lin'):
    """Plot parameters change with d_sol."""
    fig, axes = plt.subplots(3, 1, sharex='col')
    axes[0].plot(d_sol, error, 'ko')
    axes[0].axhline(error[-1], c='k', ls=':')
    axes[0].set(ylabel="Minimal error $\sigma$", title=title)  # add title if prefered
    axes[1].plot(d_sol, d_gel, 'ro')
    axes[1].axhline(np.average(d_gel), ls=':', c='r')
    axes[1].set(ylabel="D$_{gel}$ [$\mu$m$^2$/s]")
    axes[2].plot(d_sol, f_gel, 'bo')
    axes[2].axhline(np.average(f_gel), ls=':', c='b')
    axes[2].set(xlabel="D$_{bulk}$ [$\mu$m$^2$/s]",
                ylabel="$\Delta$F$_{gel}$ [k$_B$T]")
    if scale is 'log':
        for ax in axes:  # setting log scale
            ax.set(yscale='log')

    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height*1.5)  # double height because of two rows

    if save:
        plt.savefig(home+"/Desktop/%s.pdf" % name, bbox_inches='tight')
    else:
        plt.show()


def read_data(path):
    """Read data from simulations."""
    # cycle through all simulations
    D_sol, D_gel, F_gel, Error = {}, {}, {}, {}
    for set in setups:
        D_sol[set], D_gel[set], F_gel[set], Error[set] = [], [], [], []
        for d in diffusivities:
            subpath = '%s/DSol_%s/results_DSol=%.2f/' % (set, d, d)
            err_raw = np.loadtxt(path+subpath+'minError.txt', delimiter=',')
            df_raw = np.loadtxt(path+subpath+'DF_best.txt', delimiter=',')
            min_err = np.min(err_raw)  # gather minimal error
            d_sol, d_gel = df_raw[0, 1], df_raw[-1, 1]  # gather D's
            f_gel = df_raw[-1, 2]  # gather F
            # save loaded data in lists
            D_sol[set].append(d_sol)
            D_gel[set].append(d_gel)
            F_gel[set].append(f_gel)
            Error[set].append(min_err)

    return D_sol, D_gel, F_gel, Error
##########################################################################


################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
setups = ['gel10_dex10', 'gel10_dex4', 'gel6_dex10', 'gel6_dex20', 'gel6_dex4']
diffusivities = [d for d in np.arange(100, 1001, 100).astype(int)] + [1, 27.4, 39.5, 55, 101.4]
path_d_data = home+"/Desktop/Block_new_data/reflective_full_bulk/sigmoidal/"
##########################################################################


#################################
#             MAIN LOOP         #
##########################################################################
# read errors for different d values
D_sol, D_gel, F_gel, Error = read_data(path_d_data)

for i, set in enumerate(setups):
    figure_diffusivities(D_sol[set], D_gel[set], F_gel[set], Error[set],
                         title=set.split('_'), name=set, save=True)
##########################################################################
