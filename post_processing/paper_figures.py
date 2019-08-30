# -*- coding: utf-8 -*-
"""Make figures for paper."""
import numpy as np
import pandas as pd
import fitting_scripts.FPModel as fp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import mpltex  # for acs style figures


###################
#  ENVIRONMENT    #
##########################################################################
path_to_data = '/Users/woldeaman/Nextcloud/PhD/Projects/FokkerPlanckModeling/PEG_Gel/consisten_preprocessing'
##########################################################################


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
# %%
def read_data(batch, m_dex, m_gel, dt=10, path=path_to_data):
    """Read data from fit optimization."""
    # read raw data
    DF = np.loadtxt(f'{path_to_data}/{batch}.Batch/gel{m_gel}_dex{m_dex}/DF_avg.txt',
                    delimiter=',')
    cc_theo = np.loadtxt(f'{path_to_data}/{batch}.Batch/gel{m_gel}_dex{m_dex}/cc_theo_avg.txt',
                         delimiter=',')
    exp_dat = np.loadtxt(f'{path_to_data}/{batch}.Batch/gel{m_gel}_dex{m_dex}/gel{m_gel}_dex{m_dex}.txt',
                         delimiter=',')
    scalings = np.loadtxt(f'{path_to_data}/{batch}.Batch/gel{m_gel}_dex{m_dex}/scalings_avg.txt',
                          delimiter=',')
    all_results = pd.read_excel(f'{path_to_data}/{batch}.Batch/gel{m_gel}_dex{m_dex}/results.xlsx')
    # separate and compute neccessary data
    t_trans, error = all_results['Averaged Results'][3], all_results['Averaged Results'][5]
    D, D_STD, F, F_STD = DF[:, 0], DF[:, 1], DF[:, 2], DF[:, 3]
    xx, cc = exp_dat[:, 0], exp_dat[:, 1:]
    tt = np.arange(0, cc[0, :].size*dt, dt)  # time vector
    cc = fp.build_zero_profile(cc)
    cc_exp = [cc[0]]+[c*s for c, s in zip(cc[1:], scalings[:, 2])]  # scaled experimental profiles
    dxx_dist, dxx_width = fp.discretization_Block(xx)  # get variable discretization
    # build accurate xx-vector
    xx_pre = np.array([np.sum(dxx_width[i:6]) for i in range(6)])
    xx_scale = np.concatenate((xx_pre, xx))  # zero is at bin 6
    # for labeling the x-axis correctly, first 4 bins at different separation
    xx_dummy = np.concatenate(([0, 6, 12, 18], np.arange(cc_theo[:, 0].size-4)+19))
    xlabels = [np.append(xx_dummy[:3], xx_dummy[6::5]).astype(int),
               np.append(xx_scale[:3], xx_scale[6::5]).astype(int)]
    t_trans = t_trans/abs(xx[1]-xx[0]) + 19 + 2  # scale transition to new x-vector
    tt_ext = np.append(tt[:-1], np.arange(tt[-1], tt[-1]*7))  # extend to long time limit

    return (xx_dummy, xlabels, cc_exp, cc_theo, tt_ext, t_trans, D, F, D_STD, F_STD, error)


@mpltex.acs_decorator  # making acs-style figures
def figure_profiles(xx, xticks, cc_exp, cc_theo, tt, t_trans, D, F, D_STD, F_STD,
                    error, M_dex=[4, 40], plt_profiles='all', save=False,
                    savePath="/Users/woldeaman/Desktop/"):
    """Figure showing two exemplary fit results."""
    # setting number of profiles to plot
    c_nbr = [cc_theo[i][0, :].size for i in range(len(cc_theo))]  # number of numerical profiles
    plt_nbr = []
    for i in range(len(c_nbr)):
        if plt_profiles is 'all' or c_nbr[i] < plt_profiles:
            plt_nbr.append(np.arange(1, c_nbr[i]))  # plot all profiles
        else:
            # logarithmicly selecting profiles to plot, more for earlier times
            plt_nbr.append(np.unique(np.logspace(0, np.log10(c_nbr[i]-1), num=plt_profiles).astype(int)))
    # create appropriate colormap using dummy plot
    z = [[tt[i]/60, tt[i]/60, tt[i]/60] for i in range(len(tt))]  # amplitude dummy is time
    dummy_map = [plt.imshow(z[i], cmap='jet', norm=mpl.colors.LogNorm()) for i in range(len(tt))]
    # linear map between [0, 1] ~ log(t) in range of [t_min, t_max], t_min > 0
    colors = [[dummy_map[i].cmap(np.log10(tt[i][j])/(np.log10(tt[i][-1])-np.log10(tt[i][1])) -
                                 np.log10(tt[i][1])/(np.log10(tt[i][-1])-np.log10(tt[i][1])))
               for j in plt_nbr[i]] for i in range(len(tt))]

    fig = plt.figure()  # create figure
    frame1 = gridspec.GridSpec(2, 1)  # gridspec for concentration profiles
    ax_profiles_1 = fig.add_subplot(frame1[0])
    ax_profiles_2 = fig.add_subplot(frame1[1])
    frame2 = gridspec.GridSpec(2, 1)  # gridspec for 1. D(z) and F(z) profiles
    frame3 = gridspec.GridSpec(2, 1)  # gridspec for 2. D(z) and F(z) profiles
    ax_D_1 = fig.add_subplot(frame2[0])
    ax_F_1 = fig.add_subplot(frame2[1], sharex=ax_D_1)
    ax_D_2 = fig.add_subplot(frame3[0])
    ax_F_2 = fig.add_subplot(frame3[1], sharex=ax_D_2)
    ax_profiles, ax_D, ax_F = [ax_profiles_1, ax_profiles_2], [ax_D_1, ax_D_2], [ax_F_1, ax_F_2]
    # subplot labels
    fig.text(0.005, 0.96, 'A', fontsize='xx-large', weight='extra bold')  # add subplot label
    fig.text(0.66, 0.96, 'B', fontsize='xx-large', weight='extra bold')
    fig.text(0.66, 0.76, 'C', fontsize='xx-large', weight='extra bold')
    fig.text(0.005, 0.485, 'D', fontsize='xx-large', weight='extra bold')  # add subplot label
    fig.text(0.66, 0.485, 'E', fontsize='xx-large', weight='extra bold')
    fig.text(0.66, 0.28, 'F', fontsize='xx-large', weight='extra bold')

    for i in range(len(ax_profiles)):
        # creating x-vector for plotting experimental profiles
        diff = cc_theo[i][:, 1].size - cc_exp[i][1].size  # difference in lengths
        xx_exp = xx[i][diff:]  # truncated vector for plotting experimental profiles
        if i == 0:
            xx_inset = xx_exp  # store x-vector for first subplot for inset plot

        # plotting concentration profiles
        plt_c_theo, plt_c_exp = [], []
        for j, col in zip(plt_nbr[i], colors[i]):  # plot rest of profiles
            if j < len(cc_exp[i]):  # only plot experimental data if provided
                plt_c_exp.append(ax_profiles[i].plot(xx_exp, cc_exp[i][j], '.', color=col))
            plt_c_theo.append(ax_profiles[i].plot(xx[i], cc_theo[i][:, j], '--', color=col))
        ax_profiles[i].set(xlabel='z-distance [$\mu$m]', ylabel='Normalized concentration')
        plt_c_zero = ax_profiles[i].plot(xx[i], cc_exp[i][0], '-k')  # t=0 profile
        # printing legend
        if i > 0:
            ax_profiles[i].legend([plt_c_zero[0], plt_c_exp[0][0], plt_c_theo[0][0]],
                                  ["c$_{init}$ (t = 0, z)", "Experiment", "Numerical"],
                                  frameon=False, loc='lower left')
        # show also computed error
        ax_profiles[i].text(0.975, 0.9, '$\sigma$ = $\pm$ %.3f' % error[i],
                            transform=ax_profiles[i].transAxes, horizontalalignment='right')
        ax_profiles[i].text(0.975, 0.77, '$M_\\text{dex}$ = %i kDa' % M_dex[i],
                            transform=ax_profiles[i].transAxes, horizontalalignment='right')
        # create colorbar with correct labels
        fig.colorbar(dummy_map[i], label='Time [min]', pad=0.0125, ax=ax_profiles[i])

        # plotting D and F profiles
        for ax, df, df_std, col, label in zip([ax_D[i], ax_F[i]], [D[i], F[i]], [D_STD[i], F_STD[i]],
                                              ['r', 'b'], ['D [$\mu$m$^2$/s]', 'F [k$_B$T]']):
            ax.errorbar(xx[i], df, yerr=df_std, fmt='.--'+col)
            ax.axhline(df[-1], ls=':', c=col)
            ax.set_ylim([0 - 0.1*np.max(df), np.max(df) + np.max(df_std) + 0.1*np.max(df)])
            ax.set(ylabel=label)
            ax.get_yaxis().set_label_coords(-0.25, 0.5)
        ax_F[i].set(xlabel='z-distance [$\mu$m]')  # set x-axes
        plt.setp(ax_D[i].get_xticklabels(), visible=False)  # don't show x-ticks for D plot
        # indicate values in solution and in bulk
        yy_D, yy_F = [0, np.min(D[i]), np.max(D[i])], [np.min(F[i]), np.max(F[i])]
        for ax, ticks, col, form in zip([ax_F[i], ax_D[i]], [yy_F, yy_D], ['blue', 'red'], ['%.2f', '%.1f']):
            ax.set_yticks(ticks)
            ax.get_yticklabels()[-1].set_color(col)
            ax.yaxis.set_major_formatter(FormatStrFormatter(form))
        ax_D[i].get_yticklabels()[-2].set_color('red')

        # nicen up plots with background colors
        dx_2 = abs(xx[i][-2]-xx[i][-1])  # bin size in second segment
        for ax in [ax_F[i], ax_D[i], ax_profiles[i]]:
            if ax is ax_profiles[i]:
                skips = 1
            else:  # for D, F plots only use half of xticks
                skips = 2
            ax.set_xticks(xticks[i][0][::skips])
            ax.set_xticklabels(xticks[i][1][::skips])
            ax.axvline(t_trans[i], ls=':', c='k')  # indicate transition
            ax.axvspan(-2*dx_2, t_trans[i], color=[0.875, 0.875, 1], lw=0)  # bulk = blue
            ax.axvspan(t_trans[i], xx[i][-1]+2*dx_2, color=[0.9, 0.9, 0.9], lw=0)  # gel = grey
            ax.set_xlim([xx[i][0]-2*dx_2, xx[i][-1]+2*dx_2])

    # make inset for closer comparison of experimental and numerical data
    inset = inset_axes(ax_profiles_1, width="45%", height="45%", loc='lower left',
                       bbox_transform=ax_profiles_1.transAxes, bbox_to_anchor=(0.05, 0.05, 1, 1))
    for j, col in zip(plt_nbr[0], colors[0]):  # plot rest of profiles
        if j < len(cc_exp[0]):  # only plot experimental data if provided
            inset.plot(xx_inset, cc_exp[0][j], '.', color=col)
        inset.plot(xx[0], cc_theo[0][:, j], '--', color=col)
    inset.plot(xx[0], cc_exp[0][0], '-k')  # t=0 profile
    mark_inset(ax_profiles_1, inset, loc1=2, loc2=4, fc="none", ec="0.75", zorder=20)
    inset.get_yaxis().set_visible(False)
    inset.get_xaxis().set_visible(False)
    # inset.tick_params(axis='y', labelsize=6)
    # inset.set_yticks()
    # ax_profiles_1.indicate_inset_zoom(inset)
    inset.axvline(t_trans[0], ls=':', c='k')  # indicate transition
    inset.axvspan(xx[0][0], t_trans[0], color=[0.875, 0.875, 1], lw=0)  # bulk = blue
    inset.axvspan(t_trans[0], xx[0][-1], color=[0.9, 0.9, 0.9], lw=0)  # gel = grey
    inset.set_xlim([22, 35])
    inset.set_ylim([0.65, 1.02])

    # for double column figures in acs style format
    w_double = 7  # inch size for width of double column figure for ACS journals
    width, height = fig.get_size_inches()
    fig.set_size_inches(w_double, height*2)
    frame1.tight_layout(fig, rect=[0, 0, 0.67, 1], h_pad=0)
    frame2.tight_layout(fig, rect=[0.5, 0.475, 1, 1], h_pad=0)
    frame3.tight_layout(fig, rect=[0.5, 0, 1, 0.525], h_pad=0)

    if save:
        plt.savefig(savePath+'/figure_2.pdf')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def example_profiles(xx, cc, savePath='/Users/woldeaman/Desktop/'):
    """Plot example profile."""
    diff = xx.size - cc[1].size
    plt.plot(xx[diff:], cc[50], 'ko--', mfc='white')
    plt.xticks([], [])
    plt.yticks([], [])
    plt.xlabel('z-distance')
    plt.ylabel('concentration')
    plt.savefig(savePath+'/example_penetration.pdf')
# %%
##########################################################################


###############
#  MAIN LOOP  #
##########################################################################
batch, m_gel, m_dexs = 11, 10, [4, 40]  # set info for plotting
# read in data
(xx_1, xticks_1, cc_exp_1, cc_theo_1, tt_1, t_trans_1, D_1, F_1, D_STD_1, F_STD_1, error_1) = read_data(batch, m_dexs[0], m_gel, dt=10, path=path_to_data)
(xx_2, xticks_2, cc_exp_2, cc_theo_2, tt_2, t_trans_2, D_2, F_2, D_STD_2, F_STD_2, error_2) = read_data(batch, m_dexs[1], m_gel, dt=10, path=path_to_data)
# make plots
figure_profiles([xx_1, xx_2], [xticks_1, xticks_2], [cc_exp_1, cc_exp_2],
                [cc_theo_1, cc_theo_2], [tt_1, tt_2], [t_trans_1, t_trans_2],
                [D_1, D_2], [F_1, F_2], [D_STD_1, D_STD_2], [F_STD_1, F_STD_2],
                [error_1, error_2], plt_profiles=12, save=True,
                savePath="/Users/woldeaman/Desktop/")
example_profiles(xx_1, cc_exp_1)
##########################################################################
