# -*- coding: utf-8 -*-
"""Analyse D and F results of fits."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpltex  # for acs style figures


#################################
#  DEFINITIONS AND FUNCTIONS    #
##########################################################################
def read_results(path):
    """Read fit results from analysed data."""
    pd.read_excel(path+'/gel10_dex10/results.xlsx').axes

    pd.read_excel(path+'/gel10_dex10/results.xlsx').values




#################
#  ENVIRONMENT  #
##########################################################################
home = '/Users/woldeaman/'  # change home directory accordingly
gels = [6, 10]  # molecular weight of the analyzed gels [kDa]
dextrans = {6: [4, 10, 20], 10: [4, 10]}  # molecular weight of analyzed dextrans for the different gels
path_to_data = home+'/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/4.Batch/ComputedData/rescaling_live/free_DSol'
##########################################################################


###############
#  MAIN LOOP  #
##########################################################################
results = read_results(path_to_data)
