import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import FormatStrFormatter
import matplotlib as mpl
import mpltex  # for acs style figures
import os
import numpy as np
import sys


# plotting format for plots of minimal error for each transition layer distance
def plotMinError(distance, Error, ESTD, save=False,
                 path=None):
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    plt.figure()
    plt.gca().set_xlim([0, np.max(distance)])
    plt.errorbar(distance, Error, yerr=[np.zeros(ESTD.size), ESTD])
    plt.xlabel('Transition Layer Thickness d [µm]')
    plt.ylabel('Minimal Error [$\pm$ µM]')
    if save:
        plt.savefig(path+'minError.pdf', bbox_inches='tight')
    else:
        plt.show()


# plotting format for D and F in the same figure
def plotDF(xx, D, F, D_STD=None, F_STD=None, save=False, style='.',
           scale='linear', name='avgDF', path=None, xticks=None):
    """
    Plots D and F profiles
    """
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    plt.figure()
    plt.gca().set_xlim(left=xx[0])
    plt.gca().set_xlim(right=xx[-1])
    # plotting F
    if F_STD is None:
        plt.plot(xx, F, style+'b')
    else:
        plt.errorbar(xx, F, yerr=F_STD, fmt=style+'b')
    plt.ylabel('Free Energy [k$_{B}$T]', color='b')
    plt.xlabel('z-distance [µm]')
    plt.tick_params('y', colors='b')
    # plotting D
    plt.twinx()
    if D_STD is None:
        plt.plot(xx, D, style+'r')
    else:
        plt.errorbar(xx, D, yerr=D_STD, fmt=style+'r')
    # Make the y-axis label, ticks and tick labels match the line
    plt.gca().set_xlim(left=xx[0])
    plt.gca().set_xlim(right=xx[-1])
    plt.ylabel('Diffusivity [µm$^2$/s]', color='r')
    plt.yscale(scale)
    plt.tick_params('y', colors='r')
    plt.xlabel('Distance [µm]')
    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])
    if save:
        plt.savefig(path+'%s.pdf' % name, bbox_inches='tight')
    else:
        plt.show()


# for plotting concentration profiles
def plotCon(xx, cc, ccRes, tt, plt_profiles='all',
            locs=[1, 3], colorbar=False, styles=['--', '-'],
            save=False, path=None):
    """
    Plot analyzed concentration profiles.

    plt_profiles - submit number of profiles for which comparison should be plotted,
    'all' means  all profiles will be plot.
    'locs' - determines location for the two legends.
    'colorbar' - plots colorbar instead of legends.
    'styles' - defines styles for experimental and numerical profiles.
    """
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    M = cc[0, :].size  # number of profiles

    # setting number of profiles to plot
    if plt_profiles is 'all':
        plt_nbr = np.arange(M)  # go through all profiles
    else:
        skp = int(M/plt_profiles)
        plt_nbr = np.arange(0, M, skp)

    # plotting concentration profiles
    l1s = []  # for sperate legends
    l2s = []
    # mapping profiles to colormap
    lines = np.linspace(0, 1, M)
    colors = [cm.jet(x) for x in lines]
    # Set the colormap and norm
    cmap = cm.jet
    norm = mpl.colors.Normalize(vmin=tt[0]/60, vmax=tt[-1]/60)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalarMap.set_array(tt/60)  # mapping colors to time in minutes

    fig = plt.figure()
    for j in plt_nbr:
        plt.gca().set_xlim(left=xx[0])
        plt.gca().set_xlim(right=xx[-1])
        l1, = plt.plot(xx, cc[:, j], '--', color=colors[j])
        l1s.append([l1])
        if j > 0:
            # plot t=0 profile only for experiment
            # because numerical profiles are computed from this one
            l2, = plt.plot(xx, ccRes[:, j], '-', color=colors[j])
            l2s.append([l2])
    # plotting two legends, for color and linestyle
    plt.legend([l1, l2], ["Experiment", "Numerical"], loc=locs[0])
    plt.xlabel('z-distance [µm]')
    plt.ylabel('Concentration [µM]')
    # place colorbar in inset in current axis
    fig.tight_layout()
    # TODO: think about position of colorbar
    # inset = inset_axes(plt.gca(), width="40%", height="3%", loc=locs[0])
    cb1 = plt.colorbar(scalarMap, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label('Time [min]')

    if save:
        plt.savefig(path+'profiles.pdf', bbox_inches='tight')
    else:
        plt.show()


# for printing analytical solution and transition layer thicknesses
def plotConTrans(xx, cc, ccRes, c0, tt, TransIndex, layerD, save=False,
                 path=None):

    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    plt.figure()
    deltaX = abs(xx[1] - xx[0])
    M = cc[0, :].size  # number of profiles

    # ccAna = np.load('cProfiles.npy')  # change here
    # xxAna = np.linspace(0, 590.82, num=100)  # for positively charged peptide
    # xxAna = np.linspace(0, 617.91, num=100)  # for negatively charged peptide
    # indexTime = np.array([10, 500, 1000, -1])  # index for which t = 5,10,15m
    # plt.plot(xxAna, ccAna[:, indexTime[1]], 'k-.', label='Analytical')
    # plotting shaded area in transition layer and textboxes
    # conditional positions, based on distance vector
    xLeft = xx[TransIndex]-layerD/2-deltaX*1.5
    xRight = xx[TransIndex]+layerD/2-deltaX*1.5
    yMax = np.max(np.concatenate((cc, ccRes), axis=1))
    # plotting shaded region
    plt.axvspan(xLeft, xRight, color='r', lw=None, alpha=0.25)
    plt.figtext(xx[TransIndex]/np.max(xx), 0.91, 'transition layer')
    plt.text(x=xLeft-30, y=yMax, s='$D_{sol}$', va='top')
    plt.text(x=xRight+10, y=yMax, s='$D_{muc}, F_{muc}$', va='top')

    # plotting concentration profiles
    l1s = []  # for sperate legends
    l2s = []

    colors = ['r', 'm', 'c', 'b', 'y', 'k', 'g']
    for j in range(M):
        plt.gca().set_xlim(left=-deltaX)
        plt.gca().set_xlim(right=xx[-1])
        plt.xlabel('Distance [µm]')
        plt.ylabel('Concentration [µM]')
        # printing analytical solution
        # plt.plot(xxAna, ccAna[:, indexTime[j]], 'k-.')
        l1, = plt.plot(xx, cc[:, j], '--', color=colors[j],
                       label='%.2f m Experiment' % float(tt[j]/60))
        # plot computed only for t > 0, otherwise not computed
        l1s.append([l1])
        # concatenated to include constanc c0 boundary condition
        l2, = plt.plot(np.concatenate((-deltaX*np.ones(1), xx)),
                       np.concatenate((c0*np.ones(1), ccRes[:, j])),
                       '-', color=colors[j],
                       label=str(int(tt[j]/60))+'m Numerical')
        l2s.append([l2])
    # plotting two legends, for color and linestyle
    legend1 = plt.legend([l1, l2], ["Experiment", "Numerical"], loc=1)
    plt.legend([l[0] for l in l1s], ["%.2f min" % (tt[i]/60) if tt[i] % 60 != 0
                                     else "%i min" % int(tt[i]/60)
                                     for i in range(tt.size)], loc=4)
    plt.gca().add_artist(legend1)

    if save:
        plt.savefig(path+'profiles.pdf', bbox_inches='tight')
    else:
        plt.show()


# for printing c-profiles
def plotConSkin(xx, cc, ccRes, tt, locs=[0, 2], save=False, path=None,
                deltaXX=None, start=6, end=-3, xticks=None, name='profiles',
                ylabel='Concentration [µM]'):

    M = len(cc)  # number of profiles
    N = ccRes[0, :].size  # number of bins
    if deltaXX is None:
        deltaXX = np.ones(N+1)
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    # plotting concentration profiles
    l1s = []  # for sperate legends
    l2s = []
    lines = np.linspace(0, 1, M)
    colors = [cm.jet(x) for x in lines]
    # Set the colormap and norm
    cmap = cm.jet
    norm = mpl.colors.Normalize(vmin=tt[0]/60, vmax=tt[-1]/60)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalarMap.set_array(tt/60)  # mapping colors to time in minutes

    fig = plt.figure()
    for j in range(M):
        if j == 0:
            l1, = plt.plot(xx, cc[j], '--', color=colors[j])
        else:
            l1, = plt.plot(xx[start:end], cc[j], '--', color=colors[j])

        # plot computed only for t > 0, otherwise not computed
        l1s.append([l1])
        if j > 0:
            # concatenated to include constanc c0 boundary condition
            l2, = plt.plot(xx, ccRes[:, j], '-', color=colors[j])
            l2s.append([l2])
    # plotting two legends, for color and linestyle
    plt.legend([l1, l2], ["Experiment", "Numerical"], loc=locs[0],
               frameon=False)
    plt.gca().set_xlim(left=xx[0])
    plt.gca().set_xlim(right=xx[-1])
    plt.xlabel('z-distance [µm]')
    plt.ylabel(ylabel)

    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])

    # place colorbar in inset in current axis
    fig.tight_layout()
    # TODO: think about position of colorbar
    # inset = inset_axes(plt.gca(), width="40%", height="3%", loc=locs[0])
    cb1 = plt.colorbar(scalarMap, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label('Time [min]')

    if save:
        plt.savefig(path+'%s.pdf' % name, bbox_inches='tight')
    else:
        plt.show()


# for printing c-profiles
def plotBlock(xx, cc, ccRes, tt, t_sig=None, locs=[0, 2], save=False, path=None,
              plt_profiles='all', deltaXX=None, start=6, end=-3, xticks=None,
              name='profiles', ylabel='Concentration [µM]'):

    M = len(cc)  # number of profiles
    N = ccRes[0, :].size  # number of bins
    if deltaXX is None:
        deltaXX = np.ones(N+1)
    if path is None:
        if sys.platform == "darwin":  # folder for linux
            path = '/Users/AmanuelWK/Desktop/'
        elif sys.platform.startswith("linux"):  # folder for mac
            path = '/home/amanuelwk/Desktop/'

    # setting number of profiles to plot
    if plt_profiles is 'all' or M < plt_profiles:
        plt_nbr = np.arange(M)  # go through all profiles
    else:
        skp = int(M/plt_profiles)
        plt_nbr = np.arange(0, M, skp)

    # plotting concentration profiles
    l1s = []  # for sperate legends
    l2s = []
    lines = np.linspace(0, 1, M)
    colors = [cm.jet(x) for x in lines]
    # Set the colormap and norm
    cmap = cm.jet
    norm = mpl.colors.Normalize(vmin=tt[0]/60, vmax=tt[-1]/60)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalarMap.set_array(tt/60)  # mapping colors to time in minutes

    fig = plt.figure()
    for j in plt_nbr:
        if j == 0:
            l1, = plt.plot(xx, cc[j], '--', color=colors[j])
        else:
            l1, = plt.plot(xx[start:end], cc[j], '--', color=colors[j])

        # plot computed only for t > 0, otherwise not computed
        l1s.append([l1])
        if j > 0:
            # concatenated to include constanc c0 boundary condition
            l2, = plt.plot(xx, ccRes[:, j], '-', color=colors[j])
            l2s.append([l2])
    # add line indicating fitted transition
    if t_sig is not None:
        plt.axvline(t_sig, c='k', ls=':')
    # plotting two legends, for color and linestyle
    plt.legend([l1, l2], ["Experiment", "Numerical"], loc=locs[0],
               frameon=False)
    plt.gca().set_xlim(left=xx[0])
    plt.gca().set_xlim(right=xx[-1])
    plt.xlabel('z-distance [µm]')
    plt.ylabel(ylabel)

    if xticks is not None:
        plt.xticks(xticks[0], xticks[1])

    # place colorbar in inset in current axis
    fig.tight_layout()
    # TODO: think about position of colorbar
    # inset = inset_axes(plt.gca(), width="40%", height="3%", loc=locs[0])
    cb1 = plt.colorbar(scalarMap, cmap=cmap, norm=norm, orientation='vertical')
    cb1.set_label('Time [min]')

    if save:
        plt.savefig(path+'%s.pdf' % name, bbox_inches='tight')
    else:
        plt.show()


@mpltex.acs_decorator  # making acs-style figures
def plot_scalings(scalings_avg, scalings_std, c_avg_bulk, c_avg_bulk_std, tt, savePath):
    """Plot fitted average bulk concentration for each profile."""
    # make plot
    fig, axes = plt.subplots(1, 2, sharex=True)
    fig.text(0.005, 0.92, 'A', fontsize='xx-large', weight='extra bold')  # add subplot label
    fig.text(0.55, 0.92, 'B', fontsize='xx-large', weight='extra bold')
    # plot scaling factors first
    axes[0].errorbar(tt/60, scalings_avg, yerr=scalings_std, fmt='k-')
    axes[0].set_ylabel('$f_{\\text{j}}$')
    # plot average c_bulk
    axes[1].errorbar(tt/60, c_avg_bulk, yerr=c_avg_bulk_std, fmt='k-')
    axes[1].set_ylabel('$\\overline{c_{bulk}}$')
    for ax in axes:
        ax.set_xlabel('t [min]')
    plt.savefig(savePath+'scalings.pdf', bbox_inches='tight')


@mpltex.acs_decorator  # making acs-style figures
def figure_combined(xx, xticks, cc_exp, cc_theo, tt, t_trans, D, F, D_STD, F_STD,
                    error, plt_profiles='all', suffix='', save=False,
                    savePath=os.getcwd()):
    """Make nice figure for D,F profiles and concentration profiles."""
    # setting number of profiles to plot
    c_nbr = len(cc_exp)  # number of profiles
    if plt_profiles is 'all' or c_nbr < plt_profiles:
        plt_nbr = np.arange(1, c_nbr)  # plot all profiles
    else:
        skip = int(c_nbr/plt_profiles)
        plt_nbr = np.arange(skip, c_nbr, skip)
    # creating x-vector for plotting experimental profiles
    diff = cc_theo[:, 1].size - cc_exp[1].size  # difference in lengths
    xx_exp = xx[diff:]  # truncated vector for plotting experimental profiles
    # setting up the colormap and color range
    colors = [cm.jet(x) for x in np.linspace(0, 1, plt_nbr.size)]
    norm = mpl.colors.Normalize(vmin=tt[1]/60, vmax=tt[-1]/60)
    scalarMap = cm.ScalarMappable(norm=norm, cmap=cm.jet)
    scalarMap.set_array(tt[1:]/60)  # mapping colors to time in minutes

    fig = plt.figure()  # create figure
    ax_profiles = plt.subplot2grid((2, 3), (0, 0), rowspan=2, colspan=2)
    ax_D = plt.subplot2grid((2, 3), (0, 2))
    ax_F = plt.subplot2grid((2, 3), (1, 2), sharex=ax_D)
    # subplot labels
    fig.text(0.005, 0.92, 'A', fontsize='xx-large', weight='extra bold')  # add subplot label
    fig.text(0.65, 0.92, 'B', fontsize='xx-large', weight='extra bold')
    fig.text(0.65, 0.55, 'C', fontsize='xx-large', weight='extra bold')

    # plotting concentration profiles
    plt_c_zero = ax_profiles.plot(xx, cc_exp[0], '--.k')  # t=0 profile
    for j, col in zip(plt_nbr, colors):  # plot rest of profiles
        plt_c_exp = ax_profiles.plot(xx_exp, cc_exp[j], '.', color=col)
        plt_c_theo = ax_profiles.plot(xx, cc_theo[:, j], '--', color=col)
    ax_profiles.set(xlabel='z-distance [$\mu$m]', ylabel='Normalized concentration')
    # printing legend
    ax_profiles.legend([plt_c_zero[0], plt_c_exp[0], plt_c_theo[0]],
                       ["c$_{init}$ (t = 0, z)", "Experiment", "Numerical"],
                       frameon=False)
    # show also computed error
    ax_profiles.text(0.05, 0.02, '$\sigma$ = $\pm$ %.3f' % error)
    # place colorbar in inset in current axis
    fig.colorbar(scalarMap, cmap=cm.jet, norm=norm, orientation='vertical',
                 ax=ax_profiles, label='Time [min]', pad=0.0125)

    # plotting D and F profiles
    for ax, df, df_std, col, label in zip([ax_D, ax_F], [D, F], [D_STD, F_STD],
                                          ['r', 'b'], ['D [$\mu$m$^2$/s]', 'F [k$_B$T]']):
        ax.errorbar(xx, df, yerr=df_std, fmt='.--'+col)
        ax.set(ylabel=label)
        ax.get_yaxis().set_label_coords(-0.21, 0.5)
        ax.axhline(df[-1], ls=':', c=col)
        ax.set_ylim([0 - 0.1*np.max(df), np.max(df) + np.max(df_std) + 0.1*np.max(df)])
    ax_F.set(xlabel='z-distance [$\mu$m]')  # set x-axes
    plt.setp(ax_D.get_xticklabels(), visible=False)  # don't show x-ticks for D plot
    # indicate values in solution and in bulk
    yy_D, yy_F = [0, np.min(D), np.max(D)], [np.min(F), np.max(F)]
    for ax, ticks, col, form in zip([ax_F, ax_D], [yy_F, yy_D], ['blue', 'red'], ['%.2f', '%.1f']):
        ax.set_yticks(ticks)
        ax.get_yticklabels()[-1].set_color(col)
        ax.yaxis.set_major_formatter(FormatStrFormatter(form))
    ax_D.get_yticklabels()[-2].set_color('red')

    # nicen up plots with background colors
    dx_2 = abs(xx[-2]-xx[-1])  # bin size in second segment
    for ax in [ax_F, ax_D, ax_profiles]:
        if ax is ax_profiles:
            skips = 1
        else:  # for D, F plots only use half of xticks
            skips = 2
        ax.set_xticks(xticks[0][::skips])
        ax.set_xticklabels(xticks[1][::skips])
        ax.axvline(t_trans, ls=':', c='k')  # indicate transition
        ax.axvspan(-2*dx_2, t_trans, color=[0.875, 0.875, 1], lw=0)  # bulk = blue
        ax.axvspan(t_trans, xx[-1]+2*dx_2, color=[0.9, 0.9, 0.9], lw=0)  # gel = grey
        ax.set_xlim([xx[0]-2*dx_2, xx[-1]+2*dx_2])

    # for double column figures in acs style format
    w_double = 7  # inch size for width of double column figure for ACS journals
    width, height = fig.get_size_inches()
    fig.set_size_inches(w_double, height)
    fig.tight_layout(pad=0.5, w_pad=0.55)

    if save:
        plt.savefig(savePath+'results_combined_%s.eps' % suffix)
    else:
        plt.show()
