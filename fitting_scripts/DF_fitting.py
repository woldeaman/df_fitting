# -*- coding: utf-8 -*-
"""Fitting DF while also rescaling profiles"""
# use this for matplotlib on the cluster
# import matplotlib
# matplotlib.use('Agg')
import sys
import os
import numpy as np
import functools as ft
import time
import xlsxwriter as xl
import pandas as pd
import fitting_scripts.inputOutput as io
import fitting_scripts.FPModel as fp
import fitting_scripts.plottingScripts as ps
import scipy.optimize as op
import scipy.special as sp
startTime = time.time()  # start measuring run time


def save_data(xx, dxx_width, cc_scaled_best, cc_scaled_means, cc_theo_best, cc_theo_mean,
              tt_og, tt_ext, errors, t_best, t_mean, best_params, avg_params, std_params, D_mean,
              D_best, F_mean, F_best, D_std, F_std, scalings_mean, scalings_std, scalings_best,
              c_bulk_mean, c_bulk_std, c_bulk_best, nbr_runs, alpha, crit_err, savePath,
              x_tot=1780):
    """Make plots and save analyzed data."""
    # header for txt file in which concentration profiles will be saved
    header_cons = ''
    for i, t in enumerate(tt_ext):
        header_cons += ('column%i: c-profile for t_%i = %i s\n'
                        % (i+1, i, int(t)))
    # saving numerical profiles
    np.savetxt(savePath+'cc_theo_best.txt', cc_theo_best, delimiter=',',
               header='Numerically computed concentration profiles\n'+header_cons)
    np.savetxt(savePath+'cc_theo_avg.txt', cc_theo_mean, delimiter=',',
               header='Numerically computed concentration profiles\n'+header_cons)
    # saving averaged DF
    np.savetxt(savePath+'DF_avg.txt', np.c_[D_mean, D_std, F_mean-F_mean[0], F_std],
               delimiter=',',
               header=('Diffusivity and free energy profiles from analysis\n'
                       'cloumn1: average diffusivity [micro_m^2/s]\n'
                       'cloumn2: stdev of diffusivity [+/- micro_m^2/s]\n'
                       'cloumn3: average free energy [k_BT]\n'
                       'cloumn4: stdev of free energy [+/- k_BT]'))
    # saving best DF
    np.savetxt(savePath+'DF_best.txt', np.c_[D_best, F_best-F_best[0]],
               delimiter=',',
               header=('Diffusivity and free energy profiles with lowest '
                       'error from analysis\n'
                       'cloumn1: diffusivity [micro_m^2/s]\n'
                       'cloumn2: free energy [k_BT]'))

    # saving Error of top 1% of runs
    np.savetxt(savePath+'minError.txt', errors, delimiter=',',
               header=(('Minimal error averaged over %i/%i runs, ' % (errors.size, nbr_runs)) +
                       ('%i%% deviation from minimal error included.') % (crit_err*100)))
    # saving fitted average bulk concentrations
    np.savetxt(savePath+'scalings_avg.txt', np.c_[c_bulk_mean, c_bulk_std, scalings_mean, scalings_std],
               delimiter=',',
               header=('Fitted bulk concentration, averaged over all runs.\n'
                       'column1: averaged bulk concentration\n'
                       'column2: bulk concentration standart deviation\n'
                       'column3: averaged scaling coefficients\n'
                       'column2: scaling coefficients standart deviation\n'))
    np.savetxt(savePath+'scalings_best.txt', np.c_[c_bulk_best, scalings_best], delimiter=',',
               header=('Fitted bulk concentration and scaling coefficients for best run.\n'
                       'column1: bulk concentration\n'
                       'column2: scaling coefficients'))

    # build accurate xx-vector
    xx_pre = np.array([np.sum(dxx_width[i:6]) for i in range(6)])
    xx_scale = np.concatenate((xx_pre, xx))  # zero is at bin 6
    # for labeling the x-axis correctly, first 4 bins at different separation
    xx_dummy = np.concatenate(([0, 6, 12, 18], np.arange(cc_theo_best[:, 0].size-4)+19))
    xlabels = [np.append(xx_dummy[:3], xx_dummy[6::5]).astype(int),
               np.append(xx_scale[:3], xx_scale[6::5]).astype(int)]
    # plotting profiles for averaged and best parameters
    t_best = t_best/abs(xx[1]-xx[0]) + 19 + 2  # scale transition to new x-vector
    t_mean = t_mean/abs(xx[1]-xx[0]) + 19 + 2
    # compute error for averaged parameters
    residuals = np.array([c_exp - c_num[6:] for c_exp, c_num in zip(cc_scaled_means[1:],
                                                                    cc_theo_mean[:, 1:].T)])
    error_mean = np.sqrt(np.sum(residuals**2) / (cc_scaled_means[1].size *
                                                 len(cc_scaled_means[1:])))
    ps.figure_combined(xx_dummy, xlabels, cc_scaled_best, cc_theo_best, tt_ext, t_best,
                       D_best, F_best-F_best[0], np.zeros(D_best.size), np.zeros(F_best.size),
                       errors[0], plt_profiles=12, save=True, savePath=savePath, suffix='best')
    ps.figure_combined(xx_dummy, xlabels, cc_scaled_means, cc_theo_mean, tt_ext, t_mean,
                       D_mean, F_mean-F_mean[0], D_std, F_std, error_mean, plt_profiles=12, save=True,
                       savePath=savePath, suffix='avg')
    # plotting fitted average bulk concentration
    ps.plot_scalings(scalings_mean, scalings_std, c_bulk_mean, c_bulk_std, tt_og[1:],
                     save=True, savePath=savePath)

    # saving data to excel spreadsheet
    workbook = xl.Workbook(savePath+'results.xlsx')
    worksheet = workbook.add_worksheet()
    bold = workbook.add_format({'bold': True})
    # writing headers
    worksheet.write('A1', 'Parameter', bold)
    worksheet.write('B1', 'Averaged Results', bold)
    worksheet.write('C1', 'Standart Deviation', bold)
    worksheet.write('D1', 'Best Results', bold)
    worksheet.write('A2', 'D_sol [µm^2/s]', bold)
    worksheet.write('A3', 'D_gel [µm^2/s]', bold)
    worksheet.write('A4', 'F_gel [kT]', bold)
    worksheet.write('A5', 't_sig [µm]', bold)
    worksheet.write('A6', 'd_sig [µm]', bold)
    worksheet.write('A7', 'error', bold)
    if alpha > 0:  # save regularization results
        worksheet.write('A8', '||x - x_ref||', bold)
        worksheet.write('A9', '||A*x - y||', bold)
        # compute difference to solution, use best fit results
        residuals_best = np.array([c_exp - c_num[6:] for c_exp, c_num in
                                   zip(cc_scaled_best[1:], cc_theo_best[:, 1:].T)])
        err_sol = np.sqrt(np.sum(residuals_best**2) / (cc_scaled_best[1].size *  # ||A*x - y||, only scaled to physical units
                                                       len(cc_scaled_best[1:])))
        # compute regularization term, use best fit results
        err_reg = regularization_term(best_params[:2], best_params[2:4],
                                      best_params[4], best_params[5],
                                      best_params[6:], alpha=1)  # ||x - x_ref||
        worksheet.write('D8', '%.5f' % err_reg)  # write to table
        worksheet.write('D9', '%.5f' % err_sol)

    # gather original parameters
    means = [avg_params[0], avg_params[1], (avg_params[3]-avg_params[2]),
             avg_params[4], avg_params[5]]
    stdevs = [std_params[0], std_params[1], (std_params[3]+std_params[2]),
              std_params[4], std_params[5]]
    bests = [best_params[0], best_params[1], (best_params[3]-best_params[2]),
             best_params[4], best_params[5]]

    # writing entries, storing original parameters for sigmoidal curves
    for row, params in enumerate(zip(means, stdevs, bests)):
        for column, values in zip(['B', 'C', 'D'], params):
            worksheet.write('%s%i' % (column, (row+2)), '%.5f' % values)
    worksheet.write('D7', '%.5f' % np.min(errors))  # write also error
    worksheet.write('B7', '%.5f' % error_mean)

    # adjusting cell widths
    worksheet.set_column(0, 15, len('Standart Deviation'))
    workbook.close()


def average_data(result, xx, cc, crit_err):
    """
    Gather and average data from all optimization runs.

    crit_err    -   describes the percent of deviation from minimal error
                    for which results will be included in average
    """
    # used to later compute normalized error
    n_profiles = len(cc)  # number of profiles
    bins = cc[1].size  # number of bins
    combis = n_profiles-1  # number of combinations for different c-profiles

    # loading error values, factor two, because of cost function definition
    key_list = np.array(list(result.root._v_children.keys()))  # key list to easily iterate over
    error = np.array([np.sqrt(2*result[key]['cost'].values[0] / (bins*combis)) for key in key_list])
    # now determine results to include for averaging, based on distance to minimal error
    err_lim = np.min(error) + np.min(error)*crit_err  # limit in error to include for averaging
    indices = error < err_lim  # index mask for results to include

    # gathering mean for all parameters
    averages = np.mean([result[key+'/x'].values[:, 0] for key in key_list[indices]], axis=0)
    stdevs = np.std([result[key+'/x'].values[:, 0] for key in key_list[indices]], axis=0)
    best_results = result[key_list[np.argmin(error)]+'/x'].values[:, 0]

    # splitting up parameters to compute D, F profiles
    D_mean, F_mean, t_mean, d_mean = averages[:2], averages[2:4], averages[4], averages[5]
    D_std, F_std = stdevs[:2], stdevs[2:4]
    D_best, F_best, t_best, d_best = best_results[:2], best_results[2:4], best_results[4], best_results[5]

    # post processing D, F profiles
    D_mean_pre = np.array([fp.sigmoidalDF(D_mean, t_mean, d_mean, x) for x in xx])
    F_mean_pre = np.array([fp.sigmoidalDF(F_mean, t_mean, d_mean, x) for x in xx])
    D_best = np.array([fp.sigmoidalDF(D_best, t_best, d_best, x) for x in xx])
    F_best = np.array([fp.sigmoidalDF(F_best, t_best, d_best, x) for x in xx])
    segments = np.concatenate((np.zeros(6), np.arange(xx.size))).astype(int)
    D_mean, F_mean = fp.computeDF(D_mean_pre, F_mean_pre, shape=segments)
    D_best, F_best = fp.computeDF(D_best, F_best, shape=segments)

    # computing errors using error propagation for Dsol, Dmuc or Fsol, Fmuc
    # contributions of t, d neglected for now...
    DSTD_pre = np.array([np.sqrt(((0.5 - sp.erf((x-t_mean)/(np.sqrt(2)*d_mean))/2) *
                                  D_std[0])**2 + ((0.5 + sp.erf((x-t_mean)/(np.sqrt(2)*d_mean))/2) *
                                                  D_std[1])**2) for x in xx])
    FSTD_pre = np.array([np.sqrt(((0.5 - sp.erf((x-t_mean)/(np.sqrt(2)*d_mean))/2) *
                                  F_std[0])**2 + ((0.5 + sp.erf((x-t_mean)/(np.sqrt(2)*d_mean))/2) *
                                                  F_std[1])**2) for x in xx])
    # now keeping fixed stdev of D, F in first 6 bins
    DSTD, FSTD = fp.computeDF(DSTD_pre, FSTD_pre, shape=segments)
    error_sorted = np.sort(error[indices])  # sort errors for used runs

    return (best_results, averages, stdevs, F_best, D_best, t_best, d_best,
            F_mean, D_mean, t_mean, d_mean, FSTD, DSTD, error_sorted)


def cross_checking(W, cc, tt, dxx_width, dxx_dist):
    """Check numerical model for conservation of concentration."""
    # Column sum does not vanish anymore for variable binning, but equal
    # positive and negative terms appear at binning transition --> total sum vanishes
    if abs(np.sum((np.sum(W, 0)))) > 0.01:
        print("WMatrix total sum does not vanish!\nMax is:",
              np.max(np.sum(W, 0)), '\nFor each column:\n', np.sum(W, 0))
        sys.exit()

    # testing conservation of concentration
    con = np.sum(cc[0]*dxx_width)
    # compute profiles from c0 and do the same conservation check
    ccComp = [fp.calcC(cc[0], t=t, W=W) for t in tt]

    if np.any(np.array([abs(np.sum(c*dxx_width)-con)
                        for c in ccComp]) > 0.01*con):
        print('Error: Computed concentration '
              'is not conserved in profiles: \n',
              np.nonzero(np.array([abs(np.sum(c*dxx_width)-con)
                                   for c in ccComp]) > 0.01*con))
        print([np.sum(c*dxx_width) for c in ccComp], '\n')
        print('concentration:\n', con)
        print('WMatrix Size:\n', W.shape)
        print('WMatrix Row Sum:\n', np.sum(W, 0))
        print('WMatrix 2Sum:\n', np.sum(np.sum(W, 0)))
        sys.exit()


def initialize_optimization(runs, params, n_profiles, xx, DMax=1000, FMax=20):
    """Set up bounds and start values for non-linear fit."""
    # gather discretization
    dx = xx[1] - xx[0]

    # set D, F bounds
    bnds_d_up = np.ones(params)*DMax
    bnds_f_up = np.ones(params)*FMax
    bnds_d_low = np.zeros(params)
    bnds_f_low = np.ones(params)*(-FMax)
    # bounds for interface position and layer thickness zero and max x position
    bnds_td_up = np.ones(2)*np.max(xx)
    bnds_td_low = np.zeros(2)
    # bounds for scaling factors for each profile
    bnds_scale_up = np.ones(n_profiles)*100  # setting this beetwen 0-100
    bnds_scale_low = np.zeros(n_profiles)
    # setting start values
    f_init = np.zeros(params)
    d_init = (np.random.rand(runs, params)*DMax)  # randomly choose D
    td_init = np.array([50, dx*3])  # order is [t, d], set t initially to 50 µm
    scale_init = np.ones(n_profiles)  # initially no scaling
    # storing everything together
    bnds = (np.concatenate((bnds_d_low, bnds_f_low, bnds_td_low, bnds_scale_low)),
            np.concatenate((bnds_d_up, bnds_f_up, bnds_td_up, bnds_scale_up)))
    inits = [np.concatenate((d, f_init, td_init, scale_init)) for d in d_init]

    return bnds, inits


def append_result(iteration, results, idx):
    """
    Append current iteration to .hdf storage.

    iteration  -  current 'OptimizeResult' object
    results    -  .h5 storage
    idx        -  index of current iteration for storage
    """
    # first separate arrays from rest of data for storage
    is_array = {key: (True if type(val) is np.ndarray else False)
                for key, val in iteration.items()}
    # append data frame containing array-less data
    array_less = pd.DataFrame({key: iteration[key]
                               for key, val in is_array.items()
                               if not val}, index=[0])
    results.append('r%i' % idx, array_less)
    # append arrays as sub-node
    for key, val in is_array.items():
        if val:
            results.append('r%i/%s' % (idx, key), pd.DataFrame(iteration[key]))


def analysis(result, xx, cc, tt, dxx_dist, dxx_width, alpha, crit_err):
    """Analyze results from optimization runs."""
    # create new folder to save results in
    savePath = os.path.join(os.getcwd(), 'results/')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    # gather data from results objects
    (best_results, averages, stdevs, F_best, D_best, t_best, d_best,
     F_mean, D_mean, t_mean, d_mean, F_std, D_std, error) = average_data(result, xx, cc, crit_err)
    # fitted values for re-scaling concentration profiles
    scalings_mean, scalings_std, scalings_best = averages[6:], stdevs[6:], best_results[6:]

    # computing rate matrix from best and averaged results
    W_best = fp.WMatrixVar(D_best, F_best, start=4, end=None, deltaXX=dxx_dist, con=True)
    W_mean = fp.WMatrixVar(D_mean, F_mean, start=4, end=None, deltaXX=dxx_dist, con=True)
    # computing concentration profiles
    dt = abs(tt[1]-tt[0])  # get temporal discretization
    tt_ext = np.append(tt[:-1], np.arange(tt[-1], tt[-1]*7, dt))  # extend to long time limit
    cc_theo_best = np.array([fp.calcC(cc[0], (t-tt[0]), W=W_best) for t in tt_ext]).T
    cc_theo_mean = np.array([fp.calcC(cc[0], (t-tt[0]), W=W_mean) for t in tt_ext]).T

    # compute re-scaled concentration profiles
    cc_best, cc_mean = [cc[0]], [cc[0]]
    for c_b, c_m, c_og in zip(scalings_best, scalings_mean, cc[1:]):
        cc_best.append(c_og*c_b)
        cc_mean.append(c_og*c_m)
    # compute fitted average bulk concentration
    c_bulk_best = fp.compute_avg_c_bulk(cc_best, xx, dxx_width)
    c_bulk_mean = fp.compute_avg_c_bulk(cc_mean, xx, dxx_width)
    # error from gauß error propagation
    c_bulk_std = fp.compute_c_bulk_stdev(cc, scalings_std, xx)

    save_data(xx, dxx_width, cc_best, cc_mean, cc_theo_best, cc_theo_mean, tt, tt_ext,
              error, t_best, t_mean, best_results, averages, stdevs, D_mean, D_best,
              F_mean, F_best, D_std, F_std, scalings_mean, scalings_std, scalings_best,
              c_bulk_mean, c_bulk_std, c_bulk_best, result.root._v_nchildren, alpha, crit_err, savePath)


def regularization_term(d, f, t_sig, d_sig, scalings, alpha=0):
    """
    Compute regularization term for residuals.
    d/f         -   the two values for D/F in bulk/gel
    t_sig/d_sig -   transition location and width
    scalings    -   scaling parameters
    alpha       -   regularization parameter
    """
    # regularize all paramters to be as simple as possible
    sig_reg = alpha*(np.array([t_sig, d_sig]))  # transition and width is zero
    d_reg, f_reg = alpha*np.diff(d), alpha*np.diff(f)  # no change in D or F
    DF_reg = np.append(sig_reg, [f_reg, d_reg])
    scalings_reg = alpha*(scalings-1)  # scalings -> 1
    regularization = np.append(DF_reg, scalings_reg)  # regularization term
    return regularization


def resFun(parameters, xx, cc, tt, dxx_dist, dxx_width, alpha, check=False):
    """Compute residuals for non-linear optimization."""
    # separate fit parameters accordingly
    d = parameters[:2]
    f = parameters[2:4]
    t_sig, d_sig = parameters[4], parameters[5]
    scalings = parameters[6:]

    # compute sigmoidal D, F profiles
    D = np.array([fp.sigmoidalDF(d, t_sig, d_sig, x) for x in xx])
    F = np.array([fp.sigmoidalDF(f, t_sig, d_sig, x) for x in xx])
    segments = np.concatenate((np.zeros(6), np.arange(D.size))).astype(int)
    D, F = fp.computeDF(D, F, shape=segments)
    # computing WMatrix, start smaller than 6, because D, F is const. only there
    W = fp.WMatrixVar(D, F, start=4, end=None, deltaXX=dxx_dist, con=True)

    if check:  # checking for conservation of concentration
        cross_checking(W, cc, tt, dxx_width, dxx_dist)

    # compute numerical profiles
    cc_theo = [fp.calcC(cc[0], t=(t-tt[0]), W=W) for t in tt[1:]]
    # re-scale concentration profiles with fit parameters
    cc_norm = [c*norm for c, norm in zip(cc[1:], scalings)]

    # compute residual vector and reshape into one long vector
    RR = np.array([c_exp - c_num[6:] for c_exp, c_num in zip(cc_norm, cc_theo)]).T
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations
    # tykhonov regularization, only contributes for alpha > 0
    if alpha > 0:
        regularization = regularization_term(d, f, t_sig, d_sig, scalings,
                                             alpha=alpha)
        RRn = np.append(RRn, regularization)  # add regularization to residuals

    return RRn


def optimization(init, bnds, xx, cc, tt, dxx_dist, dxx_width, alpha, verbosity=0):
    """Run one iteration of the non-linear optimization."""
    # reduce residual function to one argument in order to work with algorithm
    optimize = ft.partial(resFun, xx=xx, cc=cc, tt=tt, dxx_dist=dxx_dist,
                          dxx_width=dxx_width, alpha=alpha)

    # running freely with standart termination conditions
    result = op.least_squares(optimize, init, bounds=bnds, verbose=verbosity)

    return result


def main():
    """Set up optimization and run it."""
    # reading input and setting up analysis
    verbosity, runs, ana, xx, cc, tt, alpha = io.startUp_slim()
    n_profiles = cc[0, :].size-1  # number of profiles without c(t=0)

    dxx_dist, dxx_width = fp.discretization_Block(xx)  # get variable discretization
    cc = fp.build_zero_profile(cc)  # build t=0 profile
    # set up optimization
    params = 2  # only fit here Dsol, Fsol and Dmuc, Fmuc
    bnds, inits = initialize_optimization(runs, params, n_profiles, xx)

    if ana:  # make only analysis
        print('\nDoing analysis only.')
        res = pd.HDFStore('results.h5', mode='r')
        print('Overall %i runs have been performed.' % res.root._v_nchildren)
        analysis(res, xx, cc, tt, dxx_dist, dxx_width, alpha, crit_err=0.3)
        print('\nPlots have been made and data was extraced and saved.')
        sys.exit()

    completed_runs = 1
    for i, init in enumerate(inits):  # looping through all different start values
        with pd.HDFStore('results.h5', complevel=9) as results:
            try:
                res = optimization(init, bnds, xx, cc, tt, dxx_dist, dxx_width, alpha, verbosity)
                append_result(res, results, completed_runs)  # append to .hdf storage file
                print('\nCompleted %i runs out of %i...\n' % (completed_runs, len(inits)))
                completed_runs += 1
            except KeyboardInterrupt:
                print('\n\nScript has been terminated.\nData will now be analyzed...')
                break

    results = pd.HDFStore('results.h5', mode='r')  # read storage again, now no write
    analysis(results, xx, cc, tt, dxx_dist, dxx_width, alpha, crit_err=0.3)

    return completed_runs  # returns number of runs in order to compute average time per run


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          "\nAverage time per run was %.2f minutes"
          % (((time.time() - startTime)/60),
             (time.time() - startTime)/(60*runs)))
