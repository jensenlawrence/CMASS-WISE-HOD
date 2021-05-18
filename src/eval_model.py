# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

from cobaya.model import get_model
from cobaya.run import run
import numpy as np

# ----------------------------------------------------------------------------------------------------------------------
# Model Optimization
# ----------------------------------------------------------------------------------------------------------------------

# Optimizes varied parameters of specified model
def cobaya_optimize(model_variations, likelihood_func, method, packages_path, output='', debug=False):
    """
    cobaya_optimize : ModelVariations, function, str, str, str, bool -> array-like
        Executes Cobaya's run() function.
        Optimizes the values of the parameters that are allowed to vary within the specified variation range, then
        returns these optimized values and the corresponding log-likelihood and chi-squared values.

    model_variations : ModelVariations
        The instance of the ModelVariations class containing the sampled and fixed HOD parameters.

    likelihood_func : function : floats -> float
        User-defined function for computing log-likelihoods.
        Returns a log-likelihood as a float.

    method : str
        Parameter that determines which optimizer method is used. Options are 'scipy' and 'bobyqa'.

    packages_path : str
        String representation of the absolute path to the Cobaya 'packages' directory.

    output : str
        Optional argument that sets the path to where the optimization results are saved.
        Default value is ''.
    
    debug : Bool
        Parameter that indicates whether Cobaya's debug mode should be used for more detailed console output.
        Default value is False.
    """
    info = {
        'params': model_variations.sampling_params_dict,
        'likelihood': {'my_cl_like': {'external': likelihood_func}},
        'theory': {},
        'packages_path': packages_path,
        'sampler': {'minimize': 
            {'method': method,
            'ignore_prior': False,
            'override_scipy': {'method': 'Nelder-Mead'},
            'max_evals': 1e6,
            'confidence_for_unbounded': 0.9999995}},
    }

    if bool(output):
        info['output'] = output

    if debug:
        info['debug'] = True
    
    return run(info)

# ----------------------------------------------------------------------------------------------------------------------
# Run Model MCMC Chains
# ----------------------------------------------------------------------------------------------------------------------

# Executes MCMC chains on specified model
def cobaya_mcmc(model_variations, likelihood_func, packages_path, output='', debug=False):
    """
    cobaya_mcmc : ModelVariations, function, str, str, bool -> array-like
        Executes Cobaya's run() function.
        Executes Markov Chain Monte Carlo (MCMC) chains using the parameters given by model_variations and the function
        likelihood_func to determine the posterior distribution of the HOD model parameters.

    model_variations : ModelVariations
        The instance of the ModelVariations class containing the sampled and fixed HOD parameters.

    likelihood_func : function : floats -> float
        User-defined function for computing log-likelihoods.
        Returns a log-likelihood as a float.

    packages_path : str
        String representation of the absolute path to the Cobaya 'packages' directory.

    output : str
        Optional argument that sets the path to where the MCMC results are saved.
        Default value is ''.
    
    debug : Bool
        Parameter that indicates whether Cobaya's debug mode should be used for more detailed console output.
        Default value is False.
    """
    info = {
        'params': model_variations.sampling_params_dict,
        'likelihood': {'my_cl_like': {'external': likelihood_func}},
        'theory': {},
        'packages_path': packages_path,
        'sampler': {'mcmc': 
            {'learn_proposal': True,
            'oversample': True,
            'learn_proposal_Rminus1_max': 10,
            'proposal_scale': 1.0,
            'Rminus1_stop': 0.05,
            'burn_in': '100d',
            'max_tries': '100d'}},
    }

    if bool(output):
        info['output'] = output

    if debug:
        info['debug'] = True
    
    return run(info)

# ----------------------------------------------------------------------------------------------------------------------
# CMASS Parameters Grid Search
# ----------------------------------------------------------------------------------------------------------------------

# The gridsearch function
def grid_search(likelihood_func, output, cmass_M_mins, cmass_M_1s, cmass_alphas, cmass_M_0s, cmass_sig_logms,
                wise_M_mins, wise_M_1s, wise_alphas, wise_M_0s, wise_sig_logms, R_ss, R_cs, R_sc):
    """
    grid_search : function, str, array-like, array-like, array-like, array-like, array-like, array-like, array-like,
                  array-like, array-like, array-like, array-like, array-like, float, float, float -> NoneType
        Executes a grid search on the provided parameter space to determine the maximum log-likelihood of
        likelihood_func.

    likelihood_func : function : floats -> float
        User-defined function for computing log-likelihoods.
        Returns a log-likelihood as a float.

    output : str
        Optional argument that sets the path to where the grid search results are saved.
        Default value is ''.

    cmass_M_min : float
        The minimum halo mass necessary for a CMASS dark matter halo to host a central galaxy.

    cmass_M_1 : float
        A mass parameter for CMASS satellite galaxies.

    cmass_alpha : float
        The exponent of the galaxy mass power law for CMASS galaxies.

    cmass_M_0 : float
        A mass paramter for CMASS satellite galaxies.

    cmass_sig_logm : float
        The step function smoothing parameter for CMASS dark matter halos.

    wise_M_min : float
        The minimum halo mass necessary for a WISE dark matter halo to host a central galaxy.

    wise_M_1 : float
        A mass parameter for WISE satellite galaxies.

    wise_alpha : float
        The exponent of the galaxy mass power law for WISE galaxies.

    wise_M_0 : float
        A mass paramter for WISE satellite galaxies.

    wise_sig_logm : float
        The step function smoothing parameter for WISE dark matter halos.

    R_ss : float
        The satellite-satellite correlation parameter for CMASS and WISE galaxies.

    R_cs : float
        The central-satellite correlation parameter for CMASS and WISE galaxies.

    R_sc : float
        The satellite-central correlation parameter for CMASS and WISE galaxies.
    """

    idx_dict = {
        'M_min': 0,
        'M_1': 1,
        'alpha': 2,
        'M_0': 3,
        'sig_logm': 4
    }

    counter = 0
    best_loglike = -10000
    best_cmass = np.zeros(len(idx_dict))
    best_wise = np.zeros(len(idx_dict))

    for cmass_M_min in cmass_M_mins:
        for cmass_M_1 in cmass_M_1s:
            for cmass_alpha in cmass_alphas:
                for cmass_M_0 in cmass_M_0s:
                    for cmass_sig_logm in cmass_sig_logms:
                        for wise_M_min in wise_M_mins:
                            for wise_M_1 in wise_M_1s:
                                for wise_alpha in wise_alphas:
                                    for wise_M_0 in wise_M_0s:
                                        for wise_sig_logm in wise_sig_logms:

                                            counter += 1

                                            loglike = likelihood_func(
                                                cmass_M_min = cmass_M_min,
                                                cmass_M_1 = cmass_M_1,
                                                cmass_alpha = cmass_alpha,
                                                cmass_M_0 = cmass_M_0,
                                                cmass_sig_logm = cmass_sig_logm,
                                                wise_M_min = wise_M_min,
                                                wise_M_1 = wise_M_1,
                                                wise_alpha = wise_alpha,
                                                wise_M_0 = wise_M_0,
                                                wise_sig_logm = wise_sig_logm,
                                                R_ss = R_ss,
                                                R_cs = R_cs,
                                                R_sc = R_sc
                                            )

                                            if loglike > best_loglike:
                                                best_loglike = loglike
                                                best_cmass[idx_dict['M_min']] = cmass_M_min
                                                best_cmass[idx_dict['M_1']] = cmass_M_1
                                                best_cmass[idx_dict['alpha']] = cmass_alpha
                                                best_cmass[idx_dict['M_0']] = cmass_M_0
                                                best_cmass[idx_dict['sig_logm']] = cmass_sig_logm
                                                best_wise[idx_dict['M_min']] = wise_M_min
                                                best_wise[idx_dict['M_1']] = wise_M_1
                                                best_wise[idx_dict['alpha']] = wise_alpha
                                                best_wise[idx_dict['M_0']] = wise_M_0
                                                best_wise[idx_dict['sig_logm']] = wise_sig_logm

                                                print('\n')
                                                print(f'STEP {counter}')
                                                print(f'New best log-likelihood: {best_loglike}')
                                                print(f'CMASS parameters: {best_cmass}')
                                                print(f'WISE parameters: {best_wise}')
                                                print('\n')

    print('-'*80)
    print('GRID SEARCH COMPLETE')
    print(f'Best log-likelihood: {best_loglike}')
    print(f'CMASS parameters: {best_cmass}')
    print(f'WISE parameters: {best_wise}')
    print('-'*80)

    output_file = open(f'{output}.txt', 'w')
    output_file.write(f'Log-likelihood = {best_loglike}\n')
    output_file.write(f'Chi^2 = {-2 * best_loglike}\n')
    output_file.write('\n')
    output_file.write('CMASS Parameters\n')
    output_file.write(f'M_min = {best_cmass[idx_dict["M_min"]]}\n')
    output_file.write(f'M_1 = {best_cmass[idx_dict["M_1"]]}\n')
    output_file.write(f'alpha = {best_cmass[idx_dict["alpha"]]}\n')
    output_file.write(f'M_0 = {best_cmass[idx_dict["M_0"]]}\n')
    output_file.write(f'sig_logm = {best_cmass[idx_dict["sig_logm"]]}\n')
    output_file.write('\n')
    output_file.write('WISE Parameters\n')
    output_file.write(f'M_min = {best_wise[idx_dict["M_min"]]}\n')
    output_file.write(f'M_1 = {best_wise[idx_dict["M_1"]]}\n')
    output_file.write(f'alpha = {best_wise[idx_dict["alpha"]]}\n')
    output_file.write(f'M_0 = {best_wise[idx_dict["M_0"]]}\n')
    output_file.write(f'sig_logm = {best_wise[idx_dict["sig_logm"]]}\n')
    output_file.close()

# ----------------------------------------------------------------------------------------------------------------------