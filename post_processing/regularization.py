# -*- coding: utf-8 -*-
"""Analyse runs for regularization."""
import numpy as np
import matplotlib.pyplot as plt
import mpltex  # for acs style figures

home = '/Users/woldeaman/'


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
def read_data(path, alphas):
    """Read data for all values of regularization parameter alpha."""
    err_sol, err_reg = [], []
    for a in alphas:
        data = np.loadtxt('%s/alpha_%s/res_alpha=%.6f.txt' % (path, a, a))
        err_sol.append(data[0])
        err_reg.append(data[1])

    return err_sol, err_reg


@mpltex.acs_decorator  # making acs-style figures
def plot_lcurve(alphas, err_sol, err_reg, save=False):
    """Make plot for L-Curve."""
    plt.plot(err_reg, err_sol, 'ko--')
    plt.yscale('log')
    plt.xscale('log')
    plt.xlabel('|x-x$_0$|$^2$')
    plt.ylabel('|Ax-b|$^2$')

    if save:
        plt.tight_layout()
        plt.savefig(home+'/Desktop/l_curve.pdf')
    else:
        plt.show()
##########################################################################


################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
alphas = [0.000001, 0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
path = home+"/Desktop/Cluster/jobs/fokker_planck_modelling/Block_PEG/regularization/gel10_dex70/"
##########################################################################


#################################
#             MAIN LOOP         #
##########################################################################
err_sol, err_reg = read_data(path, alphas)
plot_lcurve(alphas, err_sol, err_reg)
