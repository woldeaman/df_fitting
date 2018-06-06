"""Fitting also normalization of profiles."""
# -*- coding: utf-8 -*-
# implemented fokker-planck equation in 1D for
# use this for matplotlib on the cluster
# import matplotlib
# matplotlib.use('Agg')
import numpy as np
import time
import inputOutput as io
import FPModel as fp
import functools as ft
import scipy.optimize as op
import plottingScripts as ps
import os

startTime = time.time()

def analysis(result, xx_DF, xx, cc, df_result, tt, dx_dist, dx_width, deltaX):
    '''
    Function analyses results from ls-optimization,
    if given comparison plots of results and original concentration profiles
    cc[i, :, :] at time tt[i] will be made, where D and F is averaged over
    top 'per' percent (standart is 0.1 - averaged over top 10%)
    '''
    # ----------------- setting working parameters --------------------- #
    # saving output in current results folder in current directory
    savePath = os.path.join(os.getcwd(), 'results_scaling/')
    if not os.path.exists(savePath):
        os.makedirs(savePath)

    M = len(cc)
    # ----------------- setting working parameters --------------------- #

    # reading fitted scaling values
    c_norm = result.x
    segments = np.concatenate((np.zeros(6), np.arange(len(xx_DF)))).astype(int)
    # set previously fitted D, F profiles
    d = [df_result[0], df_result[1]]
    f = [0, df_result[2]]
    t_sig, d_sig = df_result[3], df_result[4]
    D = np.array([fp.sigmoidalDF(d, t_sig, d_sig, x) for x in xx_DF])
    F = np.array([fp.sigmoidalDF(f, t_sig, d_sig, x) for x in xx_DF])

    # now keeping fixed D, F in first 6 bins
    segments = np.concatenate((np.zeros(6), np.arange(D.size))).astype(int)
    D, F = fp.computeDF(D, F, shape=segments)

    # computing matrix with variable discretization
    # start needs to be smaller than 6, because D, F is const. only there
    W = fp.WMatrixVar(D, F, start=4, end=None, deltaXX=dx_dist, con=True)
    # computing concentration profiles for best D and F
    ccRes = np.array([fp.calcC(cc[0], tt[j], W=W) for j in range(M)]).T
    # computing scaled profiles
    cc_scale = [cc[0]]
    for c_n, c_og in zip(c_norm, cc[1:]):
        cc_scale.append(c_og*c_n)

    x_tot = 1780  # total length of system in µm
    length_bulk = x_tot - np.max(xx_DF)
    # computing fitted average bulk concentration
    c_tot = np.sum(dx_width*cc[0])  # total amount from c0 profile
    c_amount = [deltaX[1]*np.sum(c) for c in cc_scale[1:]]
    c_bulk_mean = [(c_tot - c_am)/length_bulk for c_am in c_amount]
    # -------------------------- loading results --------------------------- #

    # --------------------------- saving data ------------------------------- #
    # header for txt file in which concentration profiles will be saved
    header_cons = ''
    for i, t in enumerate(tt):
        header_cons += ('column%i: c-profile [micro_M] for t_%i = %i min\n'
                        % (i+2, i, int(t/60)))

    # saving analyzed data for best results for plotting
    np.savetxt(savePath+'profiles_scaled.txt', np.c_[xx_DF, np.array([cc[0][6:]] +
                                                                     [c for c in cc_scale[1:]]).T],
               delimiter=',',
               header=('Numerically computed concentration profiles\n'
                       'column1: x-distance [micro_m]\n'+header_cons))
    # saving best DF
    np.savetxt(savePath+'DF.txt', np.c_[xx, D, F],
               delimiter=',',
               header=('Diffusivity and free energy profiles.\n'
                       'column1: x-distance [micro_m]\n'
                       'cloumn2: diffusivity [micro_m^2/s]\n'
                       'cloumn3: free energy [k_BT]'))

    np.savetxt(savePath+'c_bulk.txt', np.c_[tt[1:], c_bulk_mean], delimiter=',',
               header=('Fitted bulk concentration\n'
                       'column1: timepoint [s]\n'
                       'cloumn2: average bulk concentration\n'))
    # --------------------------- saving data ------------------------------- #

    # ------------------------- plotting data ------------------------------- #
    # reconstruct original x-vector
    xx_og = [np.sum(dx_dist[7:i]) for i in range(6, dx_dist.size)]
    x_0 = x_tot - np.max(xx_og)
    # for labeling the x-axis correctly
    xlabels = [[xx[0]]+[x for x in xx[6::5]],
               [-x_0]+[i*5*deltaX[1] for i in range(xx[6::5].size)]]

    # plotting profiles
    t_newX_coords = int(t_sig/abs(xx_DF[0]-xx_DF[1]) + 6)
    ps.plotBlock(xx, cc_scale, ccRes, tt, t_newX_coords, locs=[1, 3],
                 save=True, path=savePath, plt_profiles=15, end=None, xticks=xlabels)
    ps.plotDF(xx, D, F, save=True, style='.--', name='DF',
              path=savePath, xticks=xlabels)
    ps.plot_average_bulk_concentration(c_bulk_mean, tt[1:], savePath)


# function for computation of residuals, given to optimization function as
# argument to be optimized
def resFun(df, df_result, cc, xx, tt, deltaX, dx_dist, dx_width):
    '''
    This function computes residuals from given D and F and Concentration
    Profiles. Additional parameters include:
    discretization width: deltaX,
    concentration at left boundary: c0,
    regularization parameter: alpha
    number of different dfParameters: dfParams
    for variable discretization -
    distance beteween bins: dx_dist
    width of individual bins: dx_width
    '''

    M = len(cc)  # number of concentration profiles, additional c0 profile

    # gathering D and F and t, d from previous fit
    d = [df_result[0], df_result[1]]
    f = [0, df_result[2]]
    t_sig, d_sig = df_result[3], df_result[4]
    D = np.array([fp.sigmoidalDF(d, t_sig, d_sig, x) for x in xx])
    F = np.array([fp.sigmoidalDF(f, t_sig, d_sig, x) for x in xx])

    # average concentration in bulk related to normalization
    cc_norm = [cc[0]]+[c*norm for c, norm in zip(cc[1:], df)]

    # now keeping fixed D, F in first 6 bins
    segments = np.concatenate((np.zeros(6), np.arange(D.size))).astype(int)
    D, F = fp.computeDF(D, F, shape=segments)

    # computing matrix with variable discretization
    # start needs to be smaller than 6, because D, F is const. only there
    W = fp.WMatrixVar(D, F, start=4, end=None, deltaXX=dx_dist, con=True)
    # compute profiles from c0 and do the same conservation check
    ccComp = [fp.calcC(cc[0], t=tt[i], W=W) for i in range(M)]

    # computing residual vector
    RR = np.array([cc_norm[j] - ccComp[j][6:] for j in range(1, M)]).T

    # calculating vector of residuals
    RRn = RR.reshape(RR.size)  # residual vector contains all deviations

    return RRn


# extra function for optimization process, written for easy parallelization
def optimization(df_result, c_bulk_range, dx_dist, dx_width, bnds,
                 cc, xx, tt, deltaX, verb=0):
    """
    Helper function for non-linear LS optimization to profiles.
    """

    optimize = ft.partial(resFun, df_result=df_result, cc=cc, xx=xx, tt=tt,
                          deltaX=deltaX, dx_dist=dx_dist, dx_width=dx_width)

    initVal = c_bulk_range
    # running freely with standart termination conditions
    result = op.least_squares(optimize, initVal, bounds=bnds,
                              max_nfev=None, verbose=verb)

    return result


def main():
    # reading input and setting up analysis
    (bc_mode, dim, verbosity, Runs, ana, deltaX, c0, xx, cc, tt, bnds, FInit,
     DInit, alpha) = io.startUp()

    # NOTE: set values from first fit here
    DSol, DGel = 55, 35.26
    FGel = 0.98
    t_sig, d_sig = 166.74, 4.47
    DF_result = [DSol, DGel, FGel, t_sig, d_sig]
    # ------------------------- discretization ------------------------ #
    # length of the different segments for computation
    x_tot = 1780  # total length of system in µm
    x_2 = np.max(xx)  # length of segment 2
    x_1 = x_tot - x_2  # length of segment 1

    # defining different discretization widths
    dx2 = deltaX  # in segment 2 and segment 3
    dx1 = (x_1-2.5*dx2)/3.5  # discretization in segment x_1
    # NOTE: discretizing segment 1 first 4 bins each at a distance of dx1
    # and next two bins with a distance between them of dx2
    deltaXX = [dx1, dx2]
    # vectors for distance between bins dxx_dist and bin width dxx_width
    # dxx_dist contains distance to previous bin, at first bin same dx is taken
    dxx_dist = np.concatenate((np.ones(4)*dx1,  # used for WMatrix
                               np.ones(2+dim+1)*dx2))
    # this vector contains width of individual bins
    dxx_width = np.concatenate((np.ones(3)*dx1, np.ones(1)*(dx1+dx2)/2,
                                np.ones(2+dim)*dx2))  # used for concentration
    # NOTE: dxx_dist has one element more than dxx_width because it for WMatrix
    # computation dx at i+1 is necccessary --> needed for last bin too
    # ------------------------- discretization ------------------------ #

    # NOTE: building c0 profile, assume c0 const. in bulk
    c_const = 1  # normalized to bulk concentration c0=1
    c0 = cc[:, 0]
    c0 = np.concatenate((np.ones(6)*c_const, c0))
    cc = [c0] + [cc[:, i] for i in range(1, cc[0, :].size)]  # now with c0
    N = len(cc)-1  # number of profiles without c0

    # only fitting scaling of profiles
    norm_c_bulk_upper = np.ones(N)*100  # scale profiles between 0-100
    norm_c_bulk_lower = np.zeros(N)
    norm_c_bulk_init = np.ones(N)  # start with no scaling
    bnds = (norm_c_bulk_lower, norm_c_bulk_upper)

    # custom x-vector, only for analysis and plotting
    xx = np.arange(c0.size)
    # used to compute sigmoidal DF profiles, x<0 for first 6 bins
    # first 6 bins have constant d, f
    xx_DF = [np.sum(dxx_dist[6:i]) for i in range(6, dxx_dist.size-1)]

    result = optimization(df_result=DF_result, c_bulk_range=norm_c_bulk_init,
                          dx_dist=dxx_dist, dx_width=dxx_width, bnds=bnds,
                          cc=cc, xx=xx_DF, tt=tt, deltaX=deltaXX, verb=verbosity)
    np.save('result.npy', np.array(result))

    # analyse result
    analysis(result, xx_DF=xx_DF, xx=xx, cc=cc, df_result=DF_result,
             tt=tt, dx_dist=dxx_dist, dx_width=dxx_width, deltaX=deltaXX)


if __name__ == "__main__":
    runs = main()
    print("\nFinished optimization!"
          "\nTotal execution time was %.2f minutes"
          % ((time.time() - startTime)/60))
