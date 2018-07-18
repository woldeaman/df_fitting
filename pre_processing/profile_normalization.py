# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
x_tot = 1780  # complete length of z-distance [µm]
path_profiles_in = "/Users/woldeaman/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/4.Batch/ExperimentalData/norm_bulk"
path_profiles_out = "/Users/woldeaman/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/4.Batch/ExperimentalData/norm_amount"
##########################################################################


############################
#        FUNCTIONS         #
##########################################################################
def normalize_to_amount(xx, profiles, dx, xx_tot=x_tot):
    """Normalize supplied profiles to total amount."""
    normalized = []  # storing normalized profiles
    xx_ex = xx_tot - np.max(xx)  # external bulk added to profile
    bins = int(xx_ex/dx)  # extending profile to normalize to total amount
    for p in profiles.T:
        c_ext = np.append(np.ones(bins)*p[0], p)  # extend to boundary
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
for set in setups:
    data = np.loadtxt('%s/%s.txt' % (path_profiles_in, set), delimiter=',')
    xx, profiles = data[:, 0], data[:, 1:]  # separate profiles and x vector
    dx = abs(xx[0] - xx[1])  # compute bin size
    # do normalization
    norm = normalize_to_amount(xx, profiles, dx)

    colors = [cm.jet(x) for x in np.linspace(0, 1, norm[:, 0].size)]

    for n, c in zip(norm.T, colors):
        plt.plot(xx, n, c=c)
    plt.show()

    # save to file
    np.savetxt('%s/%s_norm.txt' % (path_profiles_out, set), np.c_[xx, norm],
               delimiter=',')
