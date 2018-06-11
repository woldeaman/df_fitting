"""Analyse fits with different constant diffusivity."""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import mpltex  # for acs style figures

# change home directory accordingly
home = '/Users/woldeaman/'
home = '/Users/AmanuelWK/'


# script estimates experimental error and compares it to error for different
# diffusivity values in bulk
#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
@mpltex.acs_decorator  # making acs-style figures
def figure_diffusivities(d_sol, d_gel, f_gel, error, name='params_dsol',
                         title='', save=False, scale='lin', opti=None):
    """Plot parameters change with d_sol."""
    fig, axes = plt.subplots(3, 1, sharex='col')
    axes[0].plot(d_sol, error, 'ko')
    if opti is not None:
        axes[0].axvline(opti, c='k', ls=':')
    axes[0].set(ylabel="Minimal error $\sigma$", title=title)  # add title if prefered
    axes[1].plot(d_sol, d_gel, 'ro')
    if opti is not None:
        axes[1].axvline(opti, ls=':', c='r')
    axes[1].set(ylabel="D$_{gel}$ [$\mu$m$^2$/s]")
    axes[2].plot(d_sol, f_gel, 'bo')
    if opti is not None:
        axes[2].axvline(opti, ls=':', c='b')
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


def read_data(path, diffusivities):
    """Read data from simulations."""
    # cycle through all simulations
    D_sol, D_gel, F_gel, Error = {}, {}, {}, {}
    for set in setups:
        D_sol[set], D_gel[set], F_gel[set], Error[set] = [], [], [], []
        for d in diffusivities[set]:
            subpath = '%s/DSol_%s/results/' % (set, d)
            err_raw = np.loadtxt(path+subpath+'minError.txt', delimiter=',')
            df_raw = np.loadtxt(path+subpath+'DF_best.txt', delimiter=',')
            min_err = np.min(err_raw)  # gather minimal error
            d_sol, d_gel = df_raw[0, 0], df_raw[-1, 0]  # gather D's
            f_gel = df_raw[-1, 1]  # gather F
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
DSols_theo = [55, 101.4, 55, 39.5, 101.4]  # literature value for DSol for different dextrans
d_range = [d for d in range(100, 1001, 100)]  # simulated DSol values
diffusivities = {set: [1]+[d for d in range(100, 1001, 100)]+[d_theo]
                 for set, d_theo in zip(setups, DSols_theo)}
path_d_data = home+"/Desktop/Cluster/FokkerPlanckModelling/Block_Data/4.Batch/rescaling_live/fix_Dsol/"
##########################################################################


#################################
#             MAIN LOOP         #
##########################################################################
# read errors for different d values
D_sol, D_gel, F_gel, Error = read_data(path_d_data, diffusivities)


for i, set in enumerate(setups):
    figure_diffusivities(D_sol[set], D_gel[set], F_gel[set], Error[set],
                         opti=DSols_theo[i], title=set.split('_'), name=set, save=True)
##########################################################################
