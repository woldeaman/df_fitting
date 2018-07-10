"""Fit free energy and diffusivity models to measured data."""
import numpy as np
import pandas as pd
import scipy.optimize as op
import matplotlib.pyplot as plt


########################
# FUNCTION DEFINITIONS #
################################################################################
def read_results(path):
    """Read fit results from analysed data."""
    D_sol, D_gel, dF = {}, {}, {}  # storing data in dicts
    for g in gels:
        d_s, d_g, f = [], [], []
        for dex in dextrans[g]:  # cycle through all analyses
            data = pd.read_excel(path+'/gel%i_dex%i/results.xlsx' % (g, dex))
            # read and store from excel file
            parameters = np.array([data['Averaged Results'][:3].values,
                                   data['Standart Deviation'][:3].values])
            for i, store in enumerate([d_s, d_g, f]):
                store.append(parameters[:, i])  # save to list first
        D_sol[g], D_gel[g], dF[g] = np.array(d_s), np.array(d_g), np.array(f)

    return D_sol, D_gel, dF


def hydrodynamic_radius(M_w, x=0.33, a=0.46):
    """
    Compute hydrodynamic radius from molar mass.
    Based on the Mark-Houwink equation, compute the hydrodynamic radius as a
    function of the molar mass. Default values for paramters are for dextran,
    taken from Aimar et al., Journal of Membrane Science, 1990.

    x       -   pre-factor for conversion
    a       -   exponent for conversion
    M_w     -   molar mass [g/mol]
    R_h     -   hydrodynamic/Stokes radius [Å]
    """
    R_h = x * M_w**a  # based on Mark-Houwink equation and relation of R_h to gyradius

    return R_h


def partition_coefficient(r_mol, r_pore):
    """
    Compute the partition coefficient based on the molecular dimensions.
    Using the derived equation for spherical diffusing molecules and a uniform pore
    network with cylindrically shaped pores from Giddings et al., J Phys Chem, 1968.

    r_mol       -   radius of the diffusing molecule
    r_pore      -   radius of the pores
    """
    K = (1 - r_mol/r_pore)**2  # partition coefficient K = c_gel/c_bulk

    return K


def diffusivity_change(r_mol, r_pore):
    """
    Compute the reduction in diffusivity between bulk and hydrogel.
    Based on equation from Renkin, J Gen Physio, 1954, compute ratio of diffusivities
    in gel and in bulk d_gel/d_bulk.

    r_mol       -   radius of the diffusing molecule
    r_pore      -   radius of the pores
    ratio       -   ratio = d_gel/d_bulk
    """
    s = r_mol/r_pore  # ratio of the two dimensions for use in equation
    p = [-4.19, 3.87, 0, -1.372, -0.94813, 0, 2.08877, 0, -2.1444, 1]
    ratio = np.polyval(p, s)  # evaluate polynomial with coefficients from model

    return ratio
################################################################################


#################
#  ENVIRONMENT  #
################################################################################
home = '/Users/woldeaman/'  # change home directory accordingly
gels = [6, 10]  # molecular weight of the analyzed gels [kDa]
dextrans = {6: [4, 10, 20], 10: [4, 10]}  # molecular weight of analyzed dextrans for the different gels
path_to_data = home+'/Dropbox/PhD/Projects/FokkerPlanckModeling/PEG_Gel/4.Batch/ComputedData/rescaling_live/free_DSol'
################################################################################


#############
# MAIN LOOP #
################################################################################
d_sol, d_gel, df = read_results(path_to_data)  # read measured data

# compute hydrodynamic radii in Å
radii = {g: [hydrodynamic_radius(m*1000) for m in dextrans[g]] for g in gels}
################################################################################
