# -*- coding: utf-8 -*-
import numpy as np
import scipy.interpolate as ip
import matplotlib.pyplot as plt
import matplotlib.cm as cm

################################
#    SETTING UP ENVIRONMENT    #
##########################################################################
x_tot = 1780  # complete length of z-distance [µm]
path_d_data = "/Users/woldeaman/Desktop/ /"
path_profiles = "./"  # in same folder
##########################################################################

############################
#        MAIN LOOP         #
##########################################################################


# script normalizes profiles, so that total concentration remains conserved
data = np.load
