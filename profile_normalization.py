# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt
import matplotlib.cm as cm

################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
x_tot = 1780  # complete length of z-distance [µm]
path_profiles = "/Users/woldeaman/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/4.Batch/ExperimentalData/"
##########################################################################


############################
#        FUNCTIONS         #
##########################################################################
def normalize_to_amount(profiles, dx, xx_tot=x_tot):
    """Normalize supplied profiles to total amount."""
    normalized = []  # storing normalized profiles
    xx_ext = int(x_tot/dx)  # extending profile to normalize to total amount
    for p in profiles.T:
        c_ext = np.append(np.ones(xx_ext)*p[0], p)  # extend to boundary
        amount = np.sum(c_ext)  # sum up to get amount (const. binning)
        normalized.append(p/amount)  # divide to normalize
    # set bulk concentration to one again
    normalized = np.array(normalized).T/normalized[0][0]
    return normalized
##########################################################################


############################
#        MAIN LOOP         #
##########################################################################
setups = ['gel6_dex4', 'gel6_dex10', 'gel6_dex20', 'gel10_dex4', 'gel10_dex10']


# script normalizes profiles, so that total concentration remains conserved
data = np.loadtxt('%s/%s.txt' % (path_profiles, setups[4]), delimiter=',')
xx, profiles = data[:, 0], data[:, 1:]  # separate profiles and x vector
dx = abs(xx[0] - xx[1])  # compute bin size

norm = normalize_to_amount(profiles, dx)

colors = [cm.jet(x) for x in np.linspace(0, 1, norm[:, 0].size)]
for n, c in zip(norm.T, colors):
    plt.plot(xx, n, c=c)

# save to file
np.savetxt('/Users/woldeaman/Desktop/%s_norm.txt' % setups[4], np.c_[xx, norm],
           delimiter=',')
