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
from crosshod import CrossHOD
from model_variations import ModelVariations
from eval_model import cobaya_optimize, cobaya_mcmc, gridsearch
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

# Instance of CrossHOD
cmass_wise_cross_hod = CrossHOD(
    cmass_redshift_file = cmass_redshift_file,
    wise_redshift_file = wise_redshift_file,
    data_file = data_file,
    covariance_file = covariance_file,
    params_file = params_file,
    cross_hod_model = VariableCorr,
    diag_cov = False
)

# Instance of ModelVariations
cmass_wise_variations = ModelVariations(params_file)

# ----------------------------------------------------------------------------------------------------------------------
# Program Execution
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Initializing argument parser
    parser = argparse.ArgumentParser(description="""HOD model for the cross-correlation of BOSS-CMASS and WISE
                                                    galaxies at z ~ 0.5.""")
    parser.add_argument('--action', type=str, metavar='ACTION',
                        help="""Function executed by the program. Options are: optimize, mcmc, wtheta_plot,
                                posterior_plot.""")
    args = parser.parse_args()

    # Functions to be executed
    assert args.action in ('optimize', 'mcmc', 'wtheta_plot', 'posterior_plot', 'gridsearch', 'test'), 'Invalid action chosen.'

    # Running optimizer
    if args.action == 'optimize':

        # Getting output file name
        method = 'scipy'
        output = 'results/optim1'

        if output == '':
            output = input('Enter optimizer output path: ')

        # Optimization function
        cobaya_optimize(
            model_variations = cmass_wise_variations,
            likelihood_func = cmass_wise_cross_hod.nbar_likelihood,
            method = method,
            packages_path = packages_path,
            output = output,
            debug = True
        )

    # Running MCMC chains
    elif args.action == 'mcmc':

        # Getting output file name
        output = ''

        if output == '':
            output = input('Enter MCMC output path: ')

        # MCMC function
        cobaya_mcmc(
            model_variations = cmass_wise_variations,
            likelihood_func = cmass_wise_cross_hod.nbar_likelihood,
            packages_path = packages_path,
            output = output,
            debug = True
        )

    # Plotting w(theta)
    elif args.action == 'wtheta_plot':
        
        # Getting output file names
        auto_output = ''

        if auto_output == '':
            auto_output = input('Enter the autocorrelation graph output path: ')

        cross_output = ''

        if cross_output == '':
            cross_output = input('Enter the cross-correlation graph output path: ')

        # Plotting functions
        title = get_corr_title(params, cmass_wise_cross_hod.nbar_likelihood)

        cmass_autocorr_plot(
            cross_hod = cmass_wise_cross_hod,
            plot_title = title,
            save_as = auto_output
        )

        crosscorr_plot(
            cross_hod = cmass_wise_cross_hod,
            plot_title = title,
            save_as = cross_output
        )

    # Plotting MCMC posteriors
    elif args.action == 'posterior_plot':
        samples_path = input('Enter path to MCMC chain results: ')

        names = input('Enter parameter names: ')
        names = list(map(lambda x: x.strip(), names.split(',')))

        labels = input('Enter LaTeX labels for graph axes: ')
        labels = list(map(lambda x: x.strip(), labels.split(',')))

        save_as = input('Enter the graph output path: ')

        # Plotting posteriors
        posterior_plot(
            samples_path = samples_path,
            names = names,
            labels = labels,
            save_as = save_as
        )

    # CMASS parameters grid search
    elif args.action == 'gridsearch':
        likelihood_func = cmass_wise_cross_hod.nbar_likelihood
        output = 'results/grid1'
        gridsearch(params, likelihood_func, output)

    # Testing area
    elif args.action == 'test':

        likelihood = cmass_wise_cross_hod.nbar_likelihood(
            cmass_M_min = 12.94,
            cmass_M_1 = 14.15,
            cmass_alpha = 0.95,
            cmass_M_0 = 13.05,
            cmass_sig_logm = 0.43,
            wise_M_min = 13.09,
            wise_M_1 = 13.925,
            wise_alpha = 0.97,
            wise_M_0 = 12.94,
            wise_sig_logm = 0.6,
            R_ss = params["galaxy_corr"]["R_ss"]["val"],
            R_cs = params["galaxy_corr"]["R_cs"]["val"],
            R_sc = params["galaxy_corr"]["R_sc"]["val"]
        )

        print(likelihood)

# ----------------------------------------------------------------------------------------------------------------------