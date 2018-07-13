"""Check method for robustness"""
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
import mpltex  # for acs style figures


@mpltex.acs_decorator  # making acs-style figures
def plot_randomized_profiles(xx, cc, tt, save=False, savePath=os.getcwd()):
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
        plt.savefig(savePath+'random_profile.eps')
    else:
        plt.show()


# plotting profiles which were randomly scaled with factors between 1-2
path = '/Users/woldeaman/Desktop/Data/FokkerPlanckModelling/Block_Data/4.Batch/randomize_profiles/gel6_dex10/'
data = np.loadtxt(path+'gel6_dex10_random.txt', delimiter=',')
xx, cc, tt = data[:, 0], data[:, 1:], np.arange(0, data[:, 1:].size, 10)
plot_randomized_profiles(xx, cc, tt, save=True, savePath='/Users/woldeaman/Desktop/')
