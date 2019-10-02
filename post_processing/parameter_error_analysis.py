# -*- coding: utf-8 -*-
"""Analysis for permeabilty coefficients."""
import numpy as np
import pandas as pd
import fitting_scripts.FPModel as fp
import matplotlib.pyplot as plt
import mpltex  # for acs style figures


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
# %%
def read_data(path, gels, dextrans):
    """
    Read fit results and concentration profiles from desired measurement.
    path        -   path to data
    gels        -   list of molecular weight of gels [as string gel_#]
    dextrans    -   dictionary of dextrans with gels as keys {gel_#: [dex_#1, dex_#2, ...]}
    """
    results = {}   # storing data in dict
    for gl in gels:
        results[gl] = {}
        for dex in dextrans[gl]:  # cycle through all analyses
            results[gl][dex] = {}
            data = pd.read_excel(f'{path}/{gl}_{dex}/results.xlsx')
            # read and store from excel file
            parameters = np.array([data['Averaged Results'][:5].values,
                                   data['Standart Deviation'][:5].values])
            # also save scalings, profiles and discretizations
            scales = np.loadtxt(f'{path}/{gl}_{dex}/scalings_avg.txt', delimiter=',')[:, 2]
            profiles = np.loadtxt(f'{path}/{gl}_{dex}/{gl}_{dex}.txt', delimiter=',')
            xx, cc = profiles[:, 0], profiles[:, 1:]  # separate xx-vector and profiles
            cc = fp.build_zero_profile(cc)  # build c(t=0) profile
            dxx_dist, dxx_width = fp.discretization_Block(profiles[:, 0])  # get variable discretization
            # store data in dictionary
            for key, val in zip(['D_sol', 'D_gel', 'dF', 't_s', 'd_s', 'scalings', 'cc_exp', 'xx', 'dx_dist'],
                                np.hsplit(parameters, 5)+[scales, cc, xx, dxx_dist]):
                results[gl][dex][key] = val  # storing in dictionary for each dextran and gel

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


def vary_parameters(p_key, p_values, measurement, dt=10):
    """
    Vary parameters and compute error to experimental profiles.
    p_key       -   name of parameter to vary
    p_values    -   parameter values to compute error for
    measurement -   dict of parameters for chosen measurement
    """
    for p in p_values:
        measurement[p_key] = p

# def error_parameter_plot():
# %%

##########################################################################

#################
#  ENVIRONMENT  #
##########################################################################
path = "/Users/woldeaman/Nextcloud/PhD/Projects/FokkerPlanckModeling/PEG_Gel/consisten_preprocessing/"
##########################################################################

###############
#  MAIN LOOP  #
##########################################################################
res = read_data(path+'/9.Batch/', ['gel6'], {'gel6': ['dex4', 'dex10', 'dex20', 'dex40']})
res['gel6']['dex4']['D_sol']

# TODO:
# 1. read data for defined measurement
# 2. select parameter to vary and compute error for each variation
# 3. make plot of error over parameter variation
##########################################################################
