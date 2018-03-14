# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# script estimates experimental error and compares it to error for different
# diffusivity values in bulk

################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
diffusivities = np.arange(50, 1001, 50)  # analyzed diffusivity values
path_d_data = "/Users/AmanuelWK/Desktop/Block_DSol/"
path_profiles = "./"  # in same folder
##########################################################################


#################################
#             MAIN LOOP         #
##########################################################################
# read errors for different d values
error_data = [np.loadtxt("%s/results_DSol=%.2f/minError.txt" % (path_d_data, d))
              for d in diffusivities]
min_error = [np.min(e) for e in error_data]  # gather min errors

# estimate experimental error
profiles_data = np.loadtxt("%s/truncated_d.txt" % path_profiles)  # read profiles
xx, cc = profiles_data[:, 0], profiles_data[:, 1:]
# TODO: think about a way to estimate experimental error
# # first fit smoothing splines to all data points
# splines = [ip.UnivariateSpline(xx, c, s=.01) for c in cc.T]
# residuals = [s.get_residual() for s in splines]
# error_spline = np.sqrt(np.sum([r for r in residuals])/cc.size)
# pre_error_bulk = [(c[:15] - c[0])**2/c[:15].size for c in cc.T]
# error_bulk = np.sqrt(np.sum(pre_error_bulk)/cc[0, :].size)


colors = [cm.jet(x) for x in np.linspace(0, 1, cc[0, :].size)]
xx_lin = np.linspace(xx[0], xx[-1])
for c, s, col in zip(cc.T, splines, colors):
    plt.plot(xx, c, ".", c=col)
    plt.plot(xx_lin, s(xx_lin), '-', c=col)
plt.axvline(xx[15], ls='--', c='k')
plt.savefig("/Users/AmanuelWK/Desktop/test.pdf")
##########################################################################
