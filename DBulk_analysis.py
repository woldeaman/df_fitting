# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import mpltex  # for acs style figures


# script estimates experimental error and compares it to error for different
# diffusivity values in bulk
#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
@mpltex.acs_decorator  # making acs-style figures
def figure_diffusivities(d_sol, d_gel, f_gel, error, name='params_dsol',
                         save=False):
    """
    Plots other parameters change with d_sol.
    """

    fig, axes = plt.subplots(3, 1, sharex='col')
    axes[0].plot(d_sol, error, 'ko')
    axes[0].axhline(error[-1], c='k', ls=':')
    axes[0].set(ylabel="Minimal error $\sigma$")
    axes[1].plot(d_sol, d_gel, 'ro')
    axes[1].axhline(np.average(d_gel), ls=':', c='r')
    axes[1].set(ylabel="D$_{gel}$ [$\mu$m$^2$/s]")
    axes[2].plot(d_sol, f_gel, 'bo')
    axes[2].axhline(np.average(f_gel), ls=':', c='b')
    axes[2].set(xlabel="D$_{bulk}$ [$\mu$m$^2$/s]",
                ylabel="$\Delta$F$_{gel}$ [k$_B$T]")

    width, height = fig.get_size_inches()
    fig.set_size_inches(width, height*1.5)  # double height because of two rows

    if save:
        plt.savefig("/Users/AmanuelWK/Desktop/%s.pdf" % name, bbox_inches='tight')
    else:
        plt.show()
##########################################################################


################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
diffusivities = np.arange(50, 1001, 50)  # analyzed diffusivity values
path_d_data = "/Users/AmanuelWK/Desktop/BlockResults/computed_data/c0_const_bulkNormalized/varying_DSol/"
path_profiles = "./"  # in same folder
##########################################################################

#################################
#             MAIN LOOP         #
##########################################################################
# read errors for different d values
error_data = [np.loadtxt("%s/results_DSol=%.2f/minError.txt" % (path_d_data, d))
              for d in diffusivities]
min_error = [np.min(e) for e in error_data]  # gather min errors
df_values = [np.loadtxt("%s/results_DSol=%.2f/DF_best.txt"
                        % (path_d_data, d), delimiter=',') for d in diffusivities]
d_sol = [d[0, 1] for d in df_values]
d_gel, f_gel = [d[-1, 1] for d in df_values], [d[-1, 2] for d in df_values]

figure_diffusivities(d_sol, d_gel, f_gel, min_error, save=True)
##########################################################################
