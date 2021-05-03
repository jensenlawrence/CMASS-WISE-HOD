# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# Basic imports
import sys
import argparse

# Cosmology imports
from halomod.cross_correlations import HODCross

# Custom imports
sys.path.append('src')
from get_model_info import get_model_params
from crosshod import CrossHOD
from model_variations import ModelVariations
from eval_model import cobaya_optimize, cobaya_mcmc
from plot_results import wtheta_plot, posterior_plot

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
    assert args.action in ('optimize', 'mcmc', 'wtheta_plot', 'posterior_plot'), 'Invalid action chosen.'

    # Running optimizer
    if args.action == 'optimize':
        method = 'scipy'
        output = ''

        if output == '':
            output = input('Enter optimizer output path: ')

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
        output = ''

        if output == '':
            output = input('Enter MCMC output path: ')

        cobaya_mcmc(
            model_variations = cmass_wise_variations,
            likelihood_func = cmass_wise_cross_hod.nbar_likelihood,
            packages_path = packages_path,
            output = output,
            debug = True
        )

    # Plotting w(theta)
    elif args.action == 'wtheta_plot':
        cmass_s1 = r'$M_{\min} = $' + f'{params["CMASS HOD"]["M_min"]["val"]}'
        cmass_s2 = r'$M_{1} = $' + f'{params["CMASS HOD"]["M_1"]["val"]}'
        cmass_s3 = r'$\alpha = $' + f'{params["CMASS HOD"]["alpha"]["val"]}'
        cmass_s4 = r'$M_{0} = $' + f'{params["CMASS HOD"]["M_0"]["val"]}'
        cmass_s5 = r'$\sigma_{\log{M}} = $' + f'{params["CMASS HOD"]["sig_logm"]["val"]}'
        cmass_s6 = f'central = {params["CMASS HOD"]["central"]["val"]}'
        cmass_title = f'CMASS : {cmass_s1}, {cmass_s2}, {cmass_s3}, {cmass_s4}, {cmass_s5}, {cmass_s6}\n'

        wise_s1 = r'$M_{\min} = $' + f'{params["WISE HOD"]["M_min"]["val"]}'
        wise_s2 = r'$M_{1} = $' + f'{params["WISE HOD"]["M_1"]["val"]}'
        wise_s3 = r'$\alpha = $' + f'{params["WISE HOD"]["alpha"]["val"]}'
        wise_s4 = r'$M_{0} = $' + f'{params["WISE HOD"]["M_0"]["val"]}'
        wise_s5 = r'$\sigma_{\log{M}} = $' + f'{params["WISE HOD"]["sig_logm"]["val"]}'
        wise_s6 = f'central = {params["WISE HOD"]["central"]["val"]}'
        wise_title = f'WISE : {wise_s1}, {wise_s2}, {wise_s3}, {wise_s4}, {wise_s5}, {wise_s6}\n'

        R_s1 = r'$R_{ss} = $' + f'{params["galaxy_corr"]["R_ss"]["val"]}'
        R_s2 = r'$R_{cs} = $' + f'{params["galaxy_corr"]["R_cs"]["val"]}'
        R_s3 = r'$R_{sc} = $' + f'{params["galaxy_corr"]["R_sc"]["val"]}'
        R_title = f'{R_s1}, {R_s2}, {R_s3}'

        title = cmass_title + wise_title + R_title

        output = ''

        if output == '':
            output = input('Enter the graph output path: ')

        wtheta_plot(
            cross_hod = cmass_wise_cross_hod,
            plot_title = title,
            save_as = output
        )

    # Plotting MCMC posteriors
    elif args.action == 'posterior_plot':
        print('Not yet implemented.')

# ----------------------------------------------------------------------------------------------------------------------