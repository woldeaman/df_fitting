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
                         title='', save=False, scale='lin', opti=None):
    """Plot parameters change with d_sol."""
    fig, axes = plt.subplots(3, 1, sharex='col')
    axes[0].plot(d_sol, error, 'k--.')
    if opti is not None:  # plot error
        optimum = axes[0].axvline(opti, c='k', ls=':')
    axes[0].set(ylabel="Minimal error $\sigma$", title=title)  # add title if prefered
    min_err = error[np.argmin(abs(d_sol-opti))]  # minimal error
    axes[0].axhline(min_err, ls=":", c='k')  # indicate optimal error value
    axes[0].set_ylim([min_err-0.25*min_err, 2*min_err])

    # plot D_gel
    axes[1].plot(d_sol, d_gel, 'r--.')
    if opti is not None:
        axes[1].axvline(opti, ls=':', c='r')
    axes[1].set(ylabel="$D_{\\text{gel}}$ [$\mu$m$^2$/s]")
    best_dgel = d_gel[np.argmin(abs(d_sol-opti))]  # best solution for D_Gel
    axes[1].axhline(best_dgel, ls=":", c='r')  # indicate optimal value

    # plot free energy
    axes[2].plot(d_sol, f_gel, 'b--.')
    if opti is not None:
        axes[2].axvline(opti, ls=':', c='b')
    axes[2].set(xlabel="$D_{\\text{sol}}$ [$\mu$m$^2$/s]",
                ylabel="$F_{\\text{gel}}$ [$k_{\\text{B}}T$]")
    best_f = f_gel[np.argmin(abs(d_sol-opti))]  # best solution for dF
    axes[2].axhline(best_f, ls=":", c='b')  # indicate optimal value
    # legend
    axes[0].legend([optimum], ['optimum'])
    if scale is 'log':
        for ax in axes:  # setting log scale
            ax.set(yscale='log')

    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height*1.5)  # double height because of two rows

    if save:
        plt.savefig(home+"/Desktop/%s.pdf" % name, bbox_inches='tight')
    else:
        plt.show()


def read_data(path, diffusivities, sub_folder=''):
    """Read data from simulations."""
    # cycle through all simulations
    D_sol, D_gel, F_gel, Error = {}, {}, {}, {}
    for set in setups:
        D_sol[set], D_gel[set], F_gel[set], Error[set] = [], [], [], []
        for d in diffusivities[set]:
            subpath = '%s/DSol_%s/%s' % (set, d, sub_folder)
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

        # convert to sorted arrays
        order = np.argsort(D_sol[set])
        D_sol[set] = np.array(D_sol[set])[order]
        D_gel[set] = np.array(D_gel[set])[order]
        F_gel[set] = np.array(F_gel[set])[order]
        Error[set] = np.array(Error[set])[order]

    return D_sol, D_gel, F_gel, Error
##########################################################################


################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
setups = ['gel6_dex4']
DSols_theo = [333.86]  # fitted value for DSol for different dextrans
diffusivities = {set: [1, 10, 50]+[d for d in range(100, 1000, 100)]+[d_theo]
                 for set, d_theo in zip(setups, DSols_theo)}
path_d_data = home+"/Desktop/Cluster/jobs/fokkerPlanckModel/PEG_dextran/6.Batch_DSol_analysis/"
##########################################################################


#################################
#             MAIN LOOP         #
##########################################################################
# read errors for different d values
D_sol, D_gel, F_gel, Error = read_data(path_d_data, diffusivities, sub_folder='gel6_dex4/results/')

for i, set in enumerate(setups):
    figure_diffusivities(D_sol[set], D_gel[set], F_gel[set], Error[set],
                         opti=DSols_theo[i], title=set.split('_'), name=set, save=True)
##########################################################################
