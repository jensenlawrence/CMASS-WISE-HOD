# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

from cobaya.model import get_model
from cobaya.run import run
import numpy as np
from itertools import product
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Cosmology imports
from cross_correlations import HODCross
from astropy.cosmology import Planck15

# Custom imports
sys.path.append('/home/jptlawre/HODEnvironment/src')
from get_model_info import get_model_params
from cmass_wise_hod_beyond_limber import CMASS_WISE_HOD
from model_variations import ModelVariations
from eval_model import optimize_model, mcmc, gridsearch
#from plot_results import cmass_autocorr_plot, crosscorr_plot, get_corr_title, posterior_plot
#import halofit_test
import time

# ----------------------------------------------------------------------------------------------------------------------
# Packages, Data, and Parameter Paths
# ----------------------------------------------------------------------------------------------------------------------

zmin = 0.55
zmax = 0.60

packages_path = '/home/jptlawre/packages'
cmass_redshift_file = '/home/jptlawre/scratch/wca/dndz/dr12cmassN-r1-v2-flag-wted-convolved-%.2f-%.2f.txt' % (zmin, zmax)
wise_redshift_file = '/home/jptlawre/scratch/wca/dndz/blue.txt'

data_file = '/home/jptlawre/scratch/wca/data/data/combined_data_wise_wts_z%.2f_%.2f.txt' % (zmin, zmax)
covariance_file = '/home/jptlawre/scratch/wca/data/cov/combined_data_wise_wts_cov_z%.2f_%.2f_percival.txt' % (zmin, zmax)
params_file = '/home/jptlawre/scratch/wca/json/cmass_wise_params_cross_%.2f_%.2f_v3.json' % (zmin, zmax)

magbias1 = '/home/jptlawre/scratch/wca/data/magbias/unwise_DR12_cmass_zmin_%.2f_zmax_%.2f_magbias_wise_mu_cross_spec_g.txt' % (zmin, zmax)
magbias2 = '/home/jptlawre/scratch/wca/data/magbias/unwise_DR12_cmass_zmin_%.2f_zmax_%.2f_magbias_wise_g_cross_spec_mu.txt' % (zmin, zmax)
magbias3 = '/home/jptlawre/scratch/wca/data/magbias/unwise_DR12_cmass_zmin_%.2f_zmax_%.2f_magbias_mumu.txt' % (zmin, zmax)

cmass_bias_file = np.loadtxt('cmass_auto_bias_NEW.txt')
fiducial_cmass_bias = cmass_bias_file[:,2][(cmass_bias_file[:,0] < 0.5 * (zmin + zmax)) & (cmass_bias_file[:,1] > 0.5 * (zmin + zmax))]

params = get_model_params(params_file)


# ----------------------------------------------------------------------------------------------------------------------
# VariableCorr Class
# ----------------------------------------------------------------------------------------------------------------------

class VariableCorr(HODCross):
    """
    Correlation relation for constant cross-correlation pairs
    """
    R_ss = params['galaxy_corr']['R_ss']['val']
    R_cs = params['galaxy_corr']['R_cs']['val']
    R_sc = params['galaxy_corr']['R_sc']['val']

    _defaults = {"R_ss": R_ss, "R_cs": R_cs, "R_sc": R_sc}

    def R_ss(self, m):
        return self.params["R_ss"]

    def R_cs(self, m):
        return self.params["R_cs"]

    def R_sc(self, m):
        return self.params["R_sc"]

    def self_pairs(self, m):
        """
        The expected number of cross-pairs at a separation of zero
        """
        return 0  

# ----------------------------------------------------------------------------------------------------------------------
# Class Instances
# ----------------------------------------------------------------------------------------------------------------------

# Instance of CMASS_WISE_HOD
hod = CMASS_WISE_HOD(
	cmass_redshift_file = cmass_redshift_file,
	wise_redshift_file = wise_redshift_file,
	data_file = data_file,
	covariance_file = covariance_file,
	params_file = params_file,
	magbias1 = magbias1,
	magbias2 = magbias2,
	magbias3 = magbias3,
	fiducial_cmass_bias = fiducial_cmass_bias,
	cross_hod_model = VariableCorr,
	diag_covariance = False, 
	exclusion_model = 'NgMatched',
	exclusion_params = None,
	min_bin = params['min_bin']
)


# ----------------------------------------------------------------------------------------------------------------------
# Model Optimization
# ----------------------------------------------------------------------------------------------------------------------

#def optimize_model(model_variations, loglike_func, method, packages_path, output='', debug=False):
"""
Finds the values of the model CMASS-WISE HOD model parameters that minimize the log-likelihood using Cobaya's
`run` function.

Parameters
----------
model_variations : ModelVariations
	Instance of the ModelVariations class that contains the HOD parameters to be sampled and the range over which
	they are sampled, and the HOD parameters to be held fixed.
loglike_func : function
	Function that calculates the log-likelihood that the observed BOSS-CMASS and WISE data were produced by an
	HOD model with the BOSS-CMASS HOD model and WISE HOD model parameters.
method : str
	Optimization method to be used by Cobaya. Options are 'scipy' and 'bobyqa'.
packages_path : str
	String representation of the path to the Cobaya `packages` directory.
output : str, optional
	String representation of the path to where the optimization results are saved.
debug : bool, optional
	Determines whether Cobaya's debug mode should be used for more detailed console outputs.
"""

params = {'cmass_M_min': {'prior': {'min': 12,
                                                            'max': 15},
                                                  'ref':13.4665, #13.035775 , #13.035459,     #13.4779, #, #13.4779,
                                                  'latex': 'cmass_M_min'},
'cmass_M_1': {'prior': {'min': 12,
                                                            'max': 18},
                                                  'ref':   14.0044 , #14.251186, #14.1645, #, #14.1645,
                                                  'latex': 'cmass_M_1'},
'cmass_alpha': {'prior': {'min': 0.0,
                                                            'max': 2.0},
                                                  'ref': 1.1530, #0.027, #0.018273 ,#1.3566728, #1.4885, #, #1.4885,
                                                  'latex': 'cmass_alpha'},
'cmass_M_0': {'prior': {'min': 8,
                                                            'max': 15},
                                                  'ref':    13.6545, #9.4009188, #12.175, #, #12.175,
                                                  'latex': 'cmass_M_0'},
'cmass_sig_logm': {'prior': {'min': 0.1,
                                                            'max': 1.5},
                                                  'ref': 0.7681 , #0.4005, #0.351155, #0.351155, #0.287092  , #0.794, #, #0.794,
                                                  'latex': 'cmass_sig_logm'},
'wise_M_min': {'prior': {'min': 12,
                                                            'max': 15},
                                                  'ref':12.9089, #13.292, #, #13.292, #13.124484, #13.281,
                                                  'latex': 'cmass_M_min'},
'wise_M_1': {'prior': {'min': 12,
                                                            'max': 15},
                                                  'ref':    13.3915 , #13.820360843918245, #, #13.82,
                                                  'latex': 'cmass_M_1'},
'wise_alpha': {'prior': {'min': 0.5,
                                                            'max': 2.0},
                                                  'ref':    1.0776, #1.0875, # #1.0875,
                                                  'latex': 'cmass_alpha'},
'wise_M_0': {'prior': {'min': 7,
                                                            'max': 15},
                                                  'ref':    12.9118, #13.265, #, #13.265,
                                                  'latex': 'cmass_M_0'},
'wise_sig_logm': {'prior': {'min': 0.1,
                                                            'max': 1.5},
                                                  'ref':   0.9564 , #0.854, #, #0.854,
                                                  'latex': 'cmass_sig_logm'},
'R_ss': 0.0,
'R_sc': 0.0,
'R_cs': 0.0}



                                                  
# Initialize model information dictionary
info = {
	'params': params,
	'likelihood': {'my_cl_like': {'external': hod.nbar_loglike}},
	'theory': {},
	'packages_path': packages_path,
	'sampler': {'evaluate': {}}}


info['debug'] = True
run(info)

# info['params']['cmass_M_min'] = 13.4779
# info['params']['cmass_M_1'] = 14.1645
# info['params']['cmass_alpha'] = 1.4885
# info['params']['cmass_M_0'] = 12.175
# info['params']['cmass_sig_logm'] = 0.794
# info['params']['wise_M_min'] = 13.292
# info['params']['wise_M_1'] = 13.820360843918245
# info['params']['wise_alpha'] = 1.0875
# info['params']['wise_M_0'] = 13.265
# info['params']['wise_sig_logm'] = 0.854
# 
# run(info)

#Run optimizer 
#sig_logms = np.linspace(13.25,13.27, 5)
# M_0s = np.linspace(0.98, 1.02, 5)
# for M_0 in M_0s:
# 	#for sig_logm in sig_logms:
# 	print(M_0)
# 	#info['params']['wise_M_min']['ref'] = sig_logm
# 	info['params']['cmass_alpha']['ref'] = M_0
# 	try:
# 		run(info)
# 	except:
# 		continue
		

# M0s = np.linspace(13.57,13.6,5)
# for M0 in M0s:
# 	info['params']['cmass_M_min']['ref'] = M0
# 	run(info)