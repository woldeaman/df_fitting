# -*- coding: utf-8 -*-
"""Analyse runs for regularization."""
import numpy as np
import pandas as pd
import scipy.interpolate as ip
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import fitting_scripts.DF_fitting as df
import fitting_scripts.FPModel as fp
import mpltex  # for acs style figures

home = '/Users/woldeaman/'


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
def compute_exp_data(path, name):
    """Compute discretization and read experimental profiles."""
    data = np.loadtxt("{}/{}.txt".format(path, name), delimiter=',')
    xx, cc = data[:, 0], data[:, 1:]
    dxx_dist, dxx_width = fp.discretization_Block(xx)  # get variable discretization
    cc = fp.build_zero_profile(cc)  # build t=0 profile

    return cc, xx, dxx_dist, dxx_width


def read_results(path, alphas, xx, cc_all, dxx_dist, dxx_width, tt_fit,
                 subppath='alpha', dt=10):
    """Read data for all values of regularization parameter alpha."""
    err_sol, err_reg, parameters = [], [], []
    cc_exp = [cc_all[t//dt] for t in tt_fit]  # get fitted profiles only !
    for a in alphas:
        # read best fitted parameters
        res = pd.HDFStore('%s/%s_%s/results.h5' % (path, subppath, a))
        key_list = np.array(list(res.root._v_children.keys()))
        error = np.array([res[key]['cost'].values[0] for key in key_list])
        best_results = res[key_list[np.argmin(error)]+'/x'].values[:, 0]
        # compute errors
        reg_term = df.regularization_term(best_results[:2], best_results[2:4],
                                          best_results[4], best_results[5],
                                          best_results[6:], alpha=1)
        err_term = df.resFun(best_results, xx, cc_exp, tt_fit, dxx_dist, dxx_width,
                             alpha=0)
        res.close()  # always close hdf stores!
        parameters.append(best_results)
        err_sol.append(np.sqrt(np.sum(err_term**2)/(cc_exp[1].size*len(cc_exp[1:]))))
        err_reg.append(np.sum(reg_term**2))
    return err_sol, err_reg, parameters


def get_test_set_scalings(scalings, tt_fit, tt_test):
    """Interpolate fitted scaling values for test set profiles."""
    test_scalings = []
    for s in scalings:  # for each alpha interpolate scalings
        spl = ip.UnivariateSpline(tt_fit[1:], s, s=0, k=1, ext=0)
        f_test = spl(tt_test[1:])  # best guess for scalings of unfitted profiles...
        test_scalings.append(f_test)
    return test_scalings


def compute_test_performance(xx, dxx_dist, dxx_width, cc_all, parameters,
                             tt_test, test_scalings, dt=10):
    """
    Compute model performance on test set.
    xx, dxx_dist/width  -  discretization parameters
    cc_all              -  all un-scaled experimental profiles
    parameters          -  fit parameters for each alpha
    tt_test             -  times for test set
    test_scalings       -  interpolated values for scaling factors for test set
    dt                  -  time between subsequently measured experimental profiles
    """
    cc_exp = [cc_all[t//dt] for t in tt_test]  # gather test set profiles
    test_set_errs = []
    for p, s in zip(parameters, test_scalings):
        params = np.append(p[:6], s)  # parameter set for each alpha with scaling factors for test set
        residues = df.resFun(params, xx, cc_exp, tt_test, dxx_dist, dxx_width, alpha=0)
        err = np.sqrt(np.sum(residues**2)/(cc_exp[1].size * len(cc_exp[1:])))
        test_set_errs.append(err)
    return test_set_errs


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


@mpltex.acs_decorator  # making acs-style figures
def plot_scalings(scalings_fit, scalings_test, tt_fit, tt_test, alphas, save=False):
    """Make plot for scalings of different alphas."""
    fig = plt.figure()
    colors = [cm.jet(x) for x in np.linspace(0, 1, len(scalings))]
    for s_f, s_t, col in zip(scalings_fit, scalings_test, colors):
        plt.plot(np.array(tt_fit[1:])/60, s_f, '.--', c=col)
        plt.plot(np.array(tt_test[1:])/60, s_t, 's', c=col, mfc='white')
    plt.xlabel('$t_{\\text{j}}$ [min]')
    plt.ylabel('$f_{\\text{j}}$')
    # dummy plots for legend
    fit, test = plt.plot([None], '.', c=colors[-1]), plt.plot([None], 's', c=colors[-1], mfc='white')
    plt.legend([fit[0], test[0]], ['training set', 'test set'], frameon=False)
    norm = mpl.colors.Normalize(vmin=alphas[0], vmax=alphas[-1])
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    scalarMap.set_array(alphas)  # mapping colors to alpha
    # place colorbar in inset in current axis
    fig.colorbar(scalarMap, cmap=cm.jet, norm=norm, orientation='vertical',
                 ax=plt.gca(), label='$\\alpha$', pad=0.0125)

    if save:
        plt.tight_layout()
        plt.savefig(home+'/Desktop/scalings.pdf')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def plot_test_set_err(alphas, test_err, sol_err, xlabel='$\\alpha$',
                      xscale='log', save=False):
    """Make plot for performance on test set over alphas."""
    plt.plot(alphas, test_err, 'k.--', label='test set')
    plt.plot(alphas, sol_err, 'r.--', label='fitting set')
    plt.xscale(xscale)
    plt.xlabel(xlabel)
    plt.ylabel('$\sigma$')
    # legend
    plt.legend(frameon=False)

    if save:
        plt.tight_layout()
        plt.savefig(home+'/Desktop/performance.pdf')
    else:
        plt.show()
##########################################################################


################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
nfevs = [10, 25, 50, 75, 100]
# NOTE: both tt-vectors should include t = 0 s !
tt_fit = [0, 30, 60, 90, 210, 270, 300, 330, 360, 480, 510, 540, 570, 600, 630,
          660, 690, 750, 780, 810, 840, 870, 900, 930, 960, 990, 1020, 1080, 1110,
          1140, 1200, 1260, 1290, 1350, 1410, 1440, 1470, 1500, 1560, 1620, 1680,
          1770, 1800, 1860, 1980, 2010, 2040, 2070, 2100, 2160, 2190, 2220, 2310,
          2340, 2430, 2460, 2520, 2550, 2580, 2610]
tt_test = [0, 120, 150, 180, 240, 390, 420, 450, 720, 1050, 1170, 1230,
           1320, 1380, 1530, 1590, 1650, 1710, 1740, 1830, 1890, 1920, 1950,
           2130, 2250, 2280, 2370, 2400, 2490, 2640]
# alphas = ['0', '0.000001', '0.00001', '0.00005', '0.0001', '0.0005', '0.001', '0.005', '0.01', '0.05', '0.1', '0.5']
# tt_fit = [0, 30, 60, 90, 120, 180, 240, 300, 330, 360, 390, 420, 480, 510, 540, 570, 630,
#           660, 720, 750, 780, 810, 870, 900, 930, 960, 990, 1020, 1050, 1080, 1110, 1140,
#           1170, 1200, 1230, 1260, 1290, 1350, 1380, 1410, 1440, 1470, 1530, 1590, 1620,
#           1680, 1710, 1740, 1770, 1830, 1860, 1890, 1920, 1950, 1980, 2010, 2040, 2070,
#           2100, 2130, 2160, 2220, 2250, 2280, 2310, 2370, 2400, 2460, 2490, 2520, 2550,
#           2580, 2610, 2640]
# tt_test = [0, 150, 210, 270, 450, 600, 690, 840, 1320, 1500, 1560, 1650, 1800, 2190, 2340, 2430]
path = home+"/Desktop/Cluster/jobs/fokker_planck_modelling/Block_PEG/early_stopping/gel10_dex70"
##########################################################################


#################################
#             MAIN LOOP         #
##########################################################################
cc_all, xx, dxx_dist, dxx_width = compute_exp_data(path+'/fev_100/', 'gel10_dex70')
err_sol, err_reg, parameters = read_results(path, nfevs, xx, cc_all, dxx_dist,
                                            dxx_width, tt_fit, subppath='fev', dt=30)

scalings = [p[6:] for p in parameters]  # extract scaling factors from parameters
test_scalings = get_test_set_scalings(scalings, tt_fit, tt_test)
test_err = compute_test_performance(xx, dxx_dist, dxx_width, cc_all, parameters,
                                    tt_test, test_scalings, dt=30)

plot_scalings(scalings, test_scalings, tt_fit, tt_test, nfevs, save=True)
plot_lcurve(nfevs, err_sol, err_reg, save=True)
plot_test_set_err(nfevs, test_err, err_sol, xlabel='nfev', xscale='linear', save=True)
