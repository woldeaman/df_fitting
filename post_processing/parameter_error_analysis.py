# -*- coding: utf-8 -*-
"""Analysis for permeabilty coefficients."""
import numpy as np
import pandas as pd
import re
import fitting_scripts.FPModel as fp
import matplotlib.pyplot as plt
import mpltex  # for acs style figures


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
# %%
def read_data(path, gel, dex):
    """
    Read fit results and concentration profiles from desired measurement.
    path        -   path to data
    """
    results = {}
    data = pd.read_excel(f'{path}/results.xlsx')
    # read and store from excel file
    parameters = data['Averaged Results'][:6].values
    # also save scalings, profiles and discretizations
    scales = np.loadtxt(f'{path}/scalings_avg.txt', delimiter=',')[:, 2]
    profiles = np.loadtxt(f'{path}/{gel}_{dex}.txt', delimiter=',')
    xx, cc = profiles[:, 0], profiles[:, 1:]  # separate xx-vector and profiles
    cc = fp.build_zero_profile(cc)  # build c(t=0) profile
    dxx_dist, dxx_width = fp.discretization_Block(profiles[:, 0])  # get variable discretization
    # store data in dictionary
    for key, val in zip(['D_sol', 'D_gel', 'dF', 't_s', 'd_s', 'sigma', 'scalings', 'cc_exp', 'xx', 'dx_dist'],
                        list(parameters)+[scales, cc, xx, dxx_dist]):
        results[key] = val  # storing in dictionary for each dextran and gel

    return results


def compute_error(measurement, dt=10):
    """"
    Compute error for given measurement.
    measurement     -   dictionary containing all results for measurement
    dt              -   temporal discretization in seconds
    """
    # extract fit values and compute time vector
    d, f = [measurement['D_sol'], measurement['D_gel']], [0, measurement['dF']]
    t_s, d_s, xx = measurement['t_s'], measurement['d_s'], measurement['xx']
    tt = np.arange(0, len(measurement['cc_exp'])*dt, dt)

    # compute theoretical profiles
    D = np.array([fp.sigmoidalDF(d, t_s, d_s, x) for x in xx])
    F = np.array([fp.sigmoidalDF(f, t_s, d_s, x) for x in xx])
    segments = np.concatenate((np.zeros(6), np.arange(D.size))).astype(int)
    D, F = fp.computeDF(D, F, shape=segments)
    # computing WMatrix, start smaller than 6, because D, F is const. only there
    W = fp.WMatrixVar(D, F, start=4, end=None, deltaXX=measurement['dx_dist'], con=True)
    # compute numerical profiles
    cc_theo = [fp.calcC(measurement['cc_exp'][0], t=(t-tt[0]), W=W) for t in tt[1:]]
    # re-scale concentration profiles with fit parameters
    cc_norm = [c*norm for c, norm in zip(measurement['cc_exp'][1:], measurement['scalings'])]

    # compute error to experimental profiles
    residuals = np.array([c_exp - c_num[6:] for c_exp, c_num in zip(cc_norm, cc_theo)]).T
    error = np.sqrt(np.sum(residuals**2) / (cc_norm[1].size * len(cc_norm[1:])))

    return error


def parameter_sweep(p_key, p_values, measurement, dt=10):
    """
    Vary parameters and compute error to experimental profiles.
    p_key       -   name of parameter to vary
    p_values    -   parameter values to compute error for
    measurement -   dict of parameters for chosen measurement
    """
    errors = []  # gather error values for all parameter values
    for p in p_values:
        mes = measurement.copy()
        mes[p_key] = p  # change parameter
        err = compute_error(mes, dt=dt)
        errors.append(err)

    # return error for each parameter value
    sweep_data = np.c_[p_values, errors]
    return sweep_data


@mpltex.acs_decorator  # making acs-style figures
def error_parameter_plot(D_sol, D_sol_id, D_gel, D_gel_id, dF, dF_id, min_err,
                         dex=None, gel=None, name=None, save=False):
    """Make figure for parameter sweep."""
    # format correct title
    gel = int(re.findall(r'\d+', gel)[0])
    dex_m = int(re.findall(r'\d+', dex)[0])
    title = 'dPG-G%d\n$M_{\\text{dex}}$ = %d kDa' % (gel, dex_m)
    # create figure
    fig, axes = plt.subplots(1, 3, sharey=True)
    for ax, dat, opt, err, cl, lbl in zip(axes, [D_sol, D_gel, dF], [D_sol_id, D_gel_id, dF_id], [min_err]*3, ['r', 'm', 'b'],
                                          ['$D_\\text{sol}$ [$\\mu$m$^2$/s]', '$D_{\\text{gel}}$ [$\\mu$m$^2$/s]', '$\\Delta$F [k$_\\text{B}$T]']):
        ax.plot(dat[:, 0], dat[:, 1], f'.-{cl}')
        ax.axvline(opt, c=cl, ls=':')
        ax.axhline(err, c=cl, ls=':')
        ax.set(xlabel=lbl)  # add title if prefered
        ax.minorticks_on()
        ax.set_ylim([0, 0.3])

    fig.text(0.25, 0.8, title)
    axes[0].set(ylabel="$\\sigma$")
    width, height = fig.get_size_inches()
    w_double = 7  # inch size for width of double column figure for ACS journals
    fig.set_size_inches(w_double, height)  # double height because of two rows
    fig.tight_layout()

    if save:
        plt.savefig(f"/Users/woldeaman/Desktop/{name}_gel{gel}_{dex}.pdf")
    else:
        plt.show()
# %%
##########################################################################


#################
#  ENVIRONMENT  #
##########################################################################
path = "/Users/woldeaman/Nextcloud/PhD/Projects/FokkerPlanckModeling/PEG_Gel/consisten_preprocessing/"
# measured gel and dextran combinations
measurements = {9: {'gel6': ['dex4', 'dex10', 'dex20', 'dex40']},
                10: {'gel6': ['dex4', 'dex10', 'dex20', 'dex40', 'dex70'],
                     'gel10': ['dex4', 'dex4_vol2', 'dex10', 'dex10_vol2',
                               'dex20', 'dex20_vol2', 'dex40']},
                11: {'gel10': ['dex4', 'dex10', 'dex20', 'dex40']},
                12: {'gel6': ['dex10', 'dex20', 'dex40'],
                     'gel10': ['dex10', 'dex40', 'dex70']}}
##########################################################################


###############
#  MAIN LOOP  #
##########################################################################
for batch, val in measurements.items():  # loop through all measurements
    for gel, dexs in val.items():
        for dex in dexs:
            # read data
            res = read_data(f'{path}/{batch}.Batch/{gel}_{dex}/', gel, dex)
            # sweep parameter values and compute erros
            D_sol = parameter_sweep('D_sol', np.linspace(1, 500), res)
            D_gel = parameter_sweep('D_gel', np.linspace(1, 500), res)
            dF = parameter_sweep('dF', np.linspace(0, 7.5), res)
            # make plots
            error_parameter_plot(D_sol, res['D_sol'], D_gel, res['D_gel'], dF, res['dF'],
                                 res['sigma'], dex=dex, gel=gel, name=f'{batch}.Batch', save=True)
##########################################################################
