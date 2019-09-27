"""Analyse fits with different constant diffusivity."""
# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import mpltex  # for acs style figures
import re

# change home directory accordingly
home = '/Users/woldeaman/'


# script estimates experimental error and compares it to error for different
# diffusivity values in bulk
#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
# %%
@mpltex.acs_decorator  # making acs-style figures
def figure_diffusivities(d_sol, d_gel, f_gel, error, opti, name='params_dsol',
                         title='', save=False, scale='lin'):
    """Plot parameters change with d_sol."""
    # format correct title
    m_gel, m_dex = int(re.findall('\d+', title[0])[0]), int(re.findall('\d+', title[1])[0])
    title = '$M_{\\text{gel}}$ = %d kDa, $M_{\\text{dex}}$ = %d kDa' % (m_gel, m_dex)
    # create figure
    fig, axes = plt.subplots(3, 1, sharex='col')
    axes[0].plot(d_sol, error, '.k-')
    axes[0].axvline(opti, c='k', ls=':')
    axes[0].set(ylabel="Minimal error $\sigma$", title=title)  # add title if prefered
    min_err = error[np.argmin(abs(d_sol-opti))]  # minimal error
    axes[0].axhline(min_err, ls=":", c='k')  # indicate optimal error value
    axes[0].minorticks_on()
    # add inset for closer look on error profiles
    inset = inset_axes(axes[0], width="45%", height="35%", loc='lower left',
                       bbox_transform=axes[0].transAxes, bbox_to_anchor=(0.5, 0.5, 1, 1))
    mark_inset(axes[0], inset, loc1=2, loc2=4, fc="none", ec="0.75", zorder=-10)
    inset.plot(d_sol, error, '.k-')
    inset.set_ylim([axes[0].get_ylim()[0], min_err+0.25*min_err])
    inset.set_xlim([axes[0].get_xlim()[0], opti+300])
    inset.tick_params(axis='y', labelsize=6)
    inset.tick_params(axis='x', labelsize=6)
    inset.axhline(min_err, ls=":", c='k')
    inset.axvline(opti, ls=":", c='k')

    # plot D_gel
    axes[1].plot(d_sol, d_gel, '.r-')
    optimum = axes[1].axvline(opti, ls=':', c='r')
    axes[1].set(ylabel="$D_{\\text{gel}}$ [$\mu$m$^2$/s]")
    best_dgel = d_gel[np.argmin(abs(d_sol-opti))]  # best solution for D_Gel
    axes[1].axhline(best_dgel, ls=":", c='r')  # indicate optimal value
    axes[1].minorticks_on()

    # plot free energy
    axes[2].plot(d_sol, f_gel, '.b-')
    axes[2].axvline(opti, ls=':', c='b')
    axes[2].set(xlabel="$D_{\\text{sol}}$ [$\mu$m$^2$/s]",
                ylabel="$F_{\\text{gel}}$ [$k_{\\text{B}}T$]")
    best_f = f_gel[np.argmin(abs(d_sol-opti))]  # best solution for dF
    axes[2].axhline(best_f, ls=":", c='b')  # indicate optimal value
    axes[2].minorticks_on()
    # legend
    axes[1].legend([optimum], ['optimum'], frameon=False, loc='center right')
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
    D_sol, D_gel, F_gel, Error = [], [], [], []
    for d in diffusivities:
        subpath = '/DSol_%s/%s' % (d, sub_folder)
        err_raw = np.loadtxt(path+subpath+'minError.txt', delimiter=',')
        df_raw = np.loadtxt(path+subpath+'DF_best.txt', delimiter=',')
        min_err = np.min(err_raw)  # gather minimal error
        d_sol, d_gel = df_raw[0, 0], df_raw[-1, 0]  # gather D's
        f_gel = df_raw[-1, 1]  # gather F
        # save loaded data in lists
        D_sol.append(d_sol)
        D_gel.append(d_gel)
        F_gel.append(f_gel)
        Error.append(min_err)

    # convert to sorted arrays
    order = np.argsort(D_sol)
    D_sol = np.array(D_sol)[order]
    D_gel = np.array(D_gel)[order]
    F_gel = np.array(F_gel)[order]
    Error = np.array(Error)[order]

    return D_sol, D_gel, F_gel, Error
# %%
##########################################################################


################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
setups = ['gel10_dex40', 'gel6_dex40']
DSols_theo = [51.1, 13.3]  # fitted value for DSol for different setups
diffusivities = {set: [1, 50]+[d for d in range(100, 1001, 100)]+[d_theo]
                 for set, d_theo in zip(setups, DSols_theo)}
path_d_data = home+"/Desktop/Cluster/jobs/fokker_planck_modelling/Block_PEG/dsol_error_d40/12.Batch/"
##########################################################################


#################################
#             MAIN LOOP         #
##########################################################################
# read errors for different d values
D_sol, D_gel, F_gel, Error = {}, {}, {}, {}
for set in setups:
    d_sol, d_gel, f_gel, err = read_data(path_d_data+'/%s' % set, diffusivities[set],
                                         sub_folder='%s/results/' % set)
    D_sol[set], D_gel[set], F_gel[set], Error[set] = d_sol, d_gel, f_gel, err

for i, set in enumerate(setups):
    figure_diffusivities(D_sol[set], D_gel[set], F_gel[set], Error[set],
                         opti=DSols_theo[i], title=set.split('_'), name=set, save=True)
##########################################################################
