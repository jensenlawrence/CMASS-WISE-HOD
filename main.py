# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# Basic imports
import sys
import argparse
import numpy as np

# Cosmology imports
from halomod.cross_correlations import HODCross

# Custom imports
sys.path.append('src')
from get_model_info import get_model_params
from cmass_wise_hod import CMASS_WISE_HOD
from model_variations import ModelVariations
from eval_model import optimize_model, mcmc, gridsearch
from plot_results import cmass_autocorr_plot, crosscorr_plot, get_corr_title, posterior_plot

# ----------------------------------------------------------------------------------------------------------------------
# Packages, Data, and Parameter Paths
# ----------------------------------------------------------------------------------------------------------------------

packages_path = '/home/jptlawre/packages'
cmass_redshift_file = 'data/dr12cmassN.txt'
wise_redshift_file = 'data/blue.txt'
data_file = 'data/combined_data.txt'
covariance_file = 'data/combined_cov.txt'
params_file = 'param/cmass_wise_params.json'
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
noexclusion_hod = CMASS_WISE_HOD(
    cmass_redshift_file = cmass_redshift_file,
    wise_redshift_file = wise_redshift_file,
    data_file = data_file,
    covariance_file = covariance_file,
    params_file = params_file,
    cross_hod_model = VariableCorr,
    diag_covariance = False, 
    exclusion_model = None,
    exclusion_params = None
)

ngmatched_hod = CMASS_WISE_HOD(
    cmass_redshift_file = cmass_redshift_file,
    wise_redshift_file = wise_redshift_file,
    data_file = data_file,
    covariance_file = covariance_file,
    params_file = params_file,
    cross_hod_model = VariableCorr,
    diag_covariance = False, 
    exclusion_model = 'NgMatched',
    exclusion_params = None
)

# Instance of ModelVariations
cmass_wise_variations = ModelVariations(params_file)

# ----------------------------------------------------------------------------------------------------------------------
# Program Execution
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="""HOD model for the cross-correlation of BOSS-CMASS and WISE
                                                    galaxies at a redshift of z ~ 0.5.""")
    parser.add_argument('-a', '--action', type=str, metavar='action',
                        help="""Function executed by the program. Options are: optimize, mcmc, gridsearch, corr_plots,
                                posterior_plot.""")
    args = parser.parse_args()

    # Verify argument is valid
    assert args.action in ('optimize', 'mcmc', 'gridsearch', 'corr_plots',
                           'posterior_plot', 'test'), 'Invalid action chosen.'

    # Optimizer action
    if args.action == 'optimize':
        
        # Set optimizer output
        output = 'results/optim1'
        if output == '':
            output = input('Enter optimizer output path: ')
        
        # Run optimizer
        optimize_model(
            model_variations = cmass_wise_variations,
            loglike_func = ngmatched_hod.nbar_loglike,
            method = 'scipy',
            packages_path = packages_path,
            output = output,
            debug = True
        )

    # MCMC action
    elif args.action == 'mcmc':

        # Set MCMC output
        output = 'results/mcmc1'
        if output == '':
            output = input('Enter MCMC output path: ')

        # Run MCMC chains
        mcmc(
            model_variations = cmass_wise_variations,
            loglike_func = ngmatched_hod.nbar_loglike,
            packages_path = packages_path,
            output = output,
            debug = True
        )

    # Grid search action
    elif args.action == 'gridsearch':

        # Set grid search output
        output = 'results/grid1'
        if output == '':
            output = input('Enter grid search output path: ')

        # Run grid search
        gridsearch(
            params = params,
            loglike_func = ngmatched_hod.nbar_components,
            output = output
        )

    # Correlation plots action
    elif args.action == 'corr_plots':

        # Set autocorr plot output
        auto_output = ''
        if auto_output == '':
            auto_output = input('Enter autocorrelation plot output path: ')

        # Set cross-corr plot output
        cross_output = ''
        if cross_output == '':
            cross_output = input('Enter cross-correlation plot output path: ')

        # Generate correlation plots
        title = get_corr_title(params, ngmatched_hod.nbar_components)

        cmass_autocorr_plot(
            cmass_wise_hod = ngmatched_hod,
            sampled = [],
            plot_title = title,
            output = auto_output,
            dpi = 200
        )  

        crosscorr_plot(
            cmass_wise_hod = ngmatched_hod,
            sampled = [],
            plot_title = title,
            output = cross_output,
            dpi = 200
        )

    # Posterior plot action
    elif args.action == 'posterior_plot':

        # Set samples, names, and labels
        samples_path = ''
        if samples_path == '':
            samples_path = input('Enter path to MCMC chain results: ')

        names = []
        if names == []:
            names = input('Enter parameter names: ')
            names = list(map(lambda x: x.strip(), names.split(',')))

        labels = []
        if labels == []:
            labels = input('Enter LaTeX labels for graph axes: ')
            labels = list(map(lambda x: x.strip(), labels.split(',')))

        # Set posterior plot output
        output = ''
        if output == '':
            output = input('Enter posterior plot output path: ')

        # Generate posterior plot
        posterior_plot(
            samples_path = samples_path,
            names = names,
            labels = labels,
            output = output
        )

    # Test action
    elif args.action == 'test':
        print('TESTING BRANCH')

        hod_params_list = [
            (13.04, 14.0, 0.950, 13.16, 0.43, 13.09, 13.775, 0.970, 13.44, 0.60),
            (13.04, 14.0, 0.975, 13.16, 0.47, 13.09, 13.775, 0.990, 13.44, 0.58),
            (12.94, 14.1, 0.950, 13.16, 0.43, 13.19, 13.775, 1.000, 13.64, 0.60),
            (13.04, 14.0, 0.975, 13.26, 0.48, 13.09, 13.675, 1.025, 13.54, 0.55),
            (13.14, 13.9, 1.000, 13.06, 0.53, 12.99, 13.875, 1.050, 13.44, 0.65),
            (12.94, 14.1, 1.025, 13.16, 0.43, 13.19, 13.775, 0.950, 13.64, 0.60),
            (13.04, 14.0, 1.050, 13.26, 0.48, 13.09, 13.675, 0.975, 13.54, 0.55),
            (13.14, 13.9, 0.950, 13.06, 0.53, 12.99, 13.875, 1.000, 13.44, 0.65),
            (12.94, 14.1, 0.975, 13.16, 0.43, 13.19, 13.775, 1.025, 13.64, 0.60),
            (13.04, 14.0, 1.000, 13.26, 0.48, 13.09, 13.675, 1.050, 13.54, 0.55)
        ]

        # idx = 6

        for idx in range(len(hod_params_list)):

            exclusion_ngmatched = ngmatched_hod.nbar_components(
                cmass_M_min = hod_params_list[idx][0],
                cmass_M_1 = hod_params_list[idx][1],
                cmass_alpha = hod_params_list[idx][2],
                cmass_M_0 = hod_params_list[idx][3],
                cmass_sig_logm = hod_params_list[idx][4],
                wise_M_min = hod_params_list[idx][5],
                wise_M_1 = hod_params_list[idx][6],
                wise_alpha = hod_params_list[idx][7],
                wise_M_0 = hod_params_list[idx][8],
                wise_sig_logm = hod_params_list[idx][9],
                R_ss = params["galaxy_corr"]["R_ss"]["val"],
                R_cs = params["galaxy_corr"]["R_cs"]["val"],
                R_sc = params["galaxy_corr"]["R_sc"]["val"]
            )

            print(f'Set {idx + 1}: NgMatched Exclusion ' + '-'*80)
            print(f'- Autocorrelation log-likelihood: {exclusion_ngmatched["auto_loglike"]}')
            print(f'- Cross-correlation log-likelihood: {exclusion_ngmatched["cross_loglike"]}')
            print(f'- Total log-likelihood: {exclusion_ngmatched["total_loglike"]}')
            print(f'- CMASS data nbar: {exclusion_ngmatched["cmass_nbar_data"]}')
            print(f'- CMASS model nbar: {exclusion_ngmatched["cmass_nbar_model"]}')
            print(f'- CMASS nbar correction: {exclusion_ngmatched["cmass_nbar_correction"]}')
            print(f'- WISE data nbar: {exclusion_ngmatched["wise_nbar_data"]}')
            print(f'- WISE model nbar: {exclusion_ngmatched["wise_nbar_model"]}')
            print(f'- WISE nbar correction: {exclusion_ngmatched["wise_nbar_correction"]}')

# ----------------------------------------------------------------------------------------------------------------------