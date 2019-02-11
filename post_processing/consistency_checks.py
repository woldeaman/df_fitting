"""Check method for robustness"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import pandas as pd
import mpltex  # for acs style figures


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
@mpltex.acs_decorator  # making acs-style figures
def plot_profiles(xx, cc, tt, save=False, savePath=os.getcwd(), name='profiles'):
    """Plot randomized profiles."""
    fig = plt.figure()
    colors = [cm.jet(x) for x in np.linspace(0, 1, cc[0, :].size)]
    norm = mpl.colors.Normalize(vmin=tt[1]/60, vmax=tt[-1]/60)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    scalarMap.set_array(tt[1:]/60)  # mapping colors to time in minutes

    for c, col in zip(cc.T, colors):  # plot rest of profiles
        plt.plot(xx, c, '.', color=col)
    plt.xlabel('z-distance [$\mu$m]')
    plt.ylabel('Concentration')
    # place colorbar in inset in current axis
    fig.colorbar(scalarMap, cmap=cm.jet, norm=norm, orientation='vertical',
                 ax=plt.gca(), label='Time [min]', pad=0.0125)

    fig.tight_layout(pad=1, w_pad=0.55)
    if save:
        plt.savefig(savePath+'/%s.pdf' % name)
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def plot_residuals(xx, residuals, tt, t_sig, save=False, savePath=os.getcwd(),
                   name='residuals'):
    fig = plt.figure()
    colors = [cm.jet(x) for x in np.linspace(0, 1, len(residuals))]
    norm = mpl.colors.Normalize(vmin=tt[1]/60, vmax=tt[-1]/60)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    scalarMap.set_array(tt[1:]/60)  # mapping colors to time in minutes
    # compute error in bulk and in gel for comparison
    x_t = np.round(t_sig/10).astype(int)
    err_blk = np.sqrt(np.sum([res[:x_t]**2 for res in residuals])/(len(residuals)*len(xx[:x_t])))
    err_gel = np.sqrt(np.sum([res[x_t:]**2 for res in residuals])/(len(residuals)*len(xx[x_t:])))
    err_gel_txt = "$\sigma_{\\text{gel}}$ = $\pm$%i$\cdot$10$^{-3}$" % (err_gel*1000)
    err_blk_txt = "$\sigma_{\\text{sol}}$ = $\pm$%i$\cdot$10$^{-3}$" % (err_blk*1000)

    for res, col in zip(residuals, colors):  # plot residuals
        plt.plot(xx, res, 'o', color=col)
    plt.axhline(0, c='k', ls='--')
    plt.axvline(xx[x_t], c='k', ls=':')
    plt.xlabel('z-distance [$\mu$m]')
    plt.ylabel('Residuals')
    plt.gca().annotate(err_blk_txt, xy=(t_sig/2, 0), xycoords='data', xytext=(0.2, -0.25),
                       textcoords='axes fraction', bbox=dict(boxstyle="round", fc='none'),
                       arrowprops=dict(arrowstyle="-", color="k", connectionstyle="arc3,rad=0.3"),
                       horizontalalignment='right', verticalalignment='bottom')
    plt.gca().annotate(err_gel_txt, xy=(t_sig+(xx[-1]-t_sig)/2, 0), xycoords='data', xytext=(1.25, -0.25),
                       textcoords='axes fraction', bbox=dict(boxstyle="round", fc='none'),
                       arrowprops=dict(arrowstyle="-", color="k", connectionstyle="arc3,rad=-0.3"),
                       horizontalalignment='right', verticalalignment='bottom')
    # place colorbar in inset in current axis
    fig.colorbar(scalarMap, cmap=cm.jet, norm=norm, orientation='vertical',
                 ax=plt.gca(), label='Time [min]', pad=0.0125)
    fig.tight_layout(pad=0.5, w_pad=0.55)
    if save:
        plt.savefig(savePath+'%s.pdf' % name)
    else:
        plt.show()
##########################################################################


#################
#  ENVIRONMENT  #
##########################################################################
path = '/Users/woldeaman/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/8.Batch/ComputedData/'
##########################################################################


#################
#  MAIN LOOP    #
##########################################################################
def main():
    gels = [6, 10, 20]
    dextrans = {6: ['dex4', 'dex40', 'FITC'], 10: ['dex4', 'dex40', 'FITC'],
                20: ['dex4', 'dex4_cut', 'dex40']}  # dextrans measured for each gel
    dt_setups = {g: {'dex4': 10, 'dex4_cut': 10, 'dex20': 10, 'dex40': 10, 'FITC': 10, 'dex70': 30} for g in gels}
    for g in gels:
        for d in dextrans[g]:
            # plotting profiles
            path_p = path+'/gel%i_%s' % (g, d)  # path to experimental profiles
            data = np.loadtxt(path_p+'/gel%i_%s.txt' % (g, d), delimiter=',')  # read profile data
            dt = dt_setups[g][d]  # intervall between recorded stacks
            xx, cc_exp, tt = data[:, 0], data[:, 1:], np.arange(0, dt*data[0, 1:].size, dt)
            # plot_profiles(xx, cc_exp, tt, save=True, savePath='/Users/woldeaman/Desktop/', name='dex4_solution')
            # plotting residuals
            path_res = path+'/gel%i_%s' % (g, d)
            t_sig = pd.read_excel(path_res+'/results.xlsx')['Averaged Results'].values[3]
            scalings = np.loadtxt(path_res+'/scalings_best.txt', delimiter=',')[:, 1]
            cc_scaled = [f*c for f, c in zip(scalings, cc_exp.T)]  # compute scaled experimental data
            cc_theo = np.loadtxt(path_res+'/cc_theo_best.txt', delimiter=',')
            residuals = [c_e - c_t[6:] for c_e, c_t in zip(cc_scaled[1:], cc_theo[:, 1:].T)]
            plot_residuals(xx, residuals, tt, t_sig, save=True, savePath='/Users/woldeaman/Desktop/', name='gel%s_%s_residuals' % (g, d))


if __name__ == "__main_":
    main()
##########################################################################
