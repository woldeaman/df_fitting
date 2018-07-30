"""Check method for robustness"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import mpltex  # for acs style figures


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
@mpltex.acs_decorator  # making acs-style figures
def plot_profiles(xx, cc, tt, save=False, savePath=os.getcwd()):
    """Plot randomized profiles."""
    fig = plt.figure()
    colors = [cm.jet(x) for x in np.linspace(0, 1, len(cc))]
    norm = mpl.colors.Normalize(vmin=tt[1]/60, vmax=tt[-1]/60)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    scalarMap.set_array(tt[1:]/60)  # mapping colors to time in minutes

    for c, col in zip(cc.T, colors):  # plot rest of profiles
        plt.plot(xx, c, '-', color=col)
    plt.xlabel('z-distance [$\mu$m]')
    plt.ylabel('Normalized concentration')
    # place colorbar in inset in current axis
    fig.colorbar(scalarMap, cmap=cm.jet, norm=norm, orientation='vertical',
                 ax=plt.gca(), label='Time [min]', pad=0.0125)

    fig.tight_layout(pad=0.5, w_pad=0.55)
    if save:
        plt.savefig(savePath+'profiles.eps')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def plot_residuals(xx, residuals, tt, save=False, savePath=os.getcwd()):
    fig = plt.figure()
    colors = [cm.jet(x) for x in np.linspace(0, 1, len(residuals))]
    norm = mpl.colors.Normalize(vmin=tt[1]/60, vmax=tt[-1]/60)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    scalarMap.set_array(tt[1:]/60)  # mapping colors to time in minutes

    for res, col in zip(residuals, colors):  # plot residuals
        plt.plot(xx, res, 'o', color=col)
    plt.axhline(0, c='k', ls=':')
    plt.xlabel('z-distance [$\mu$m]')
    plt.ylabel('Residuals')
    # place colorbar in inset in current axis
    fig.colorbar(scalarMap, cmap=cm.jet, norm=norm, orientation='vertical',
                 ax=plt.gca(), label='Time [min]', pad=0.0125)
    fig.tight_layout(pad=0.5, w_pad=0.55)
    if save:
        plt.savefig(savePath+'residuals.eps')
    else:
        plt.show()
##########################################################################


#################
#  MAIN LOOP    #
##########################################################################
# plotting profiles
path_p = ''  # path to experimental profiles
data = np.loadtxt(path_p, delimiter=',')  # read profile data
xx, cc_exp, tt = data[:, 0], data[:, 1:], np.arange(0, data[:, 1:].size, 10)
plot_profiles(xx, cc_exp, tt, save=True, savePath='/Users/woldeaman/Desktop/')
# plotting residuals
path_res = ''  # path to fit results
scalings = np.loadtxt(path_res+'/scalings_best.txt', delimiter=',')[:, 1]
cc_scaled = [f*c for f, c in zip(scalings, cc_exp.T)]  # compute scaled experimental data
cc_theo = np.loadtxt(path_res+'/cc_theo_best.txt', delimiter=',')
residuals = [c_e - c_t[6:] for c_e, c_t in zip(cc_scaled[1:], cc_theo[:, 1:].T)]
plot_residuals(xx, residuals, tt, save=True, savePath='/Users/woldeaman/Desktop/')
##########################################################################
