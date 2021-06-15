# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

from cobaya.model import get_model
from cobaya.run import run
import numpy as np
from itertools import product 

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
# CMASS-WISE Parameters Grid Search
# ----------------------------------------------------------------------------------------------------------------------

# Getting grid search parameter ranges
def get_gridsearch_range(params, key, param):
    """
    Docstring goes here
    """
    if params[key][param]['sample']:
        param_vals = np.linspace(
            params[key][param]['sample_min'],
            params[key][param]['sample_max'],
            params[key][param]['sample_div']
        )
    else:
        param_vals = [params[key][param]['val']]

    return param_vals

# Executing grid search
def gridsearch(params, likelihood_func, output):
    """
    Docstring goes here
    """
    cmass_M_min_vals = get_gridsearch_range(params, 'CMASS HOD', 'M_min')
    cmass_M_1_vals = get_gridsearch_range(params, 'CMASS HOD', 'M_1')
    cmass_alpha_vals = get_gridsearch_range(params, 'CMASS HOD', 'alpha')
    cmass_M_0_vals = get_gridsearch_range(params, 'CMASS HOD', 'M_0')
    cmass_sig_logm_vals = get_gridsearch_range(params, 'CMASS HOD', 'sig_logm')
    wise_M_min_vals = get_gridsearch_range(params, 'WISE HOD', 'M_min')
    wise_M_1_vals = get_gridsearch_range(params, 'WISE HOD', 'M_1')
    wise_alpha_vals = get_gridsearch_range(params, 'WISE HOD', 'alpha')
    wise_M_0_vals = get_gridsearch_range(params, 'WISE HOD', 'M_0')
    wise_sig_logm_vals = get_gridsearch_range(params, 'WISE HOD', 'sig_logm')
    R_ss = params['galaxy_corr']['R_ss']['val']
    R_cs = params['galaxy_corr']['R_cs']['val']
    R_sc = params['galaxy_corr']['R_sc']['val']

    print('CMASS')
    print(f'M_min = {cmass_M_min_vals}')
    print(f'M_1 = {cmass_M_1_vals}')
    print(f'alpha = {cmass_alpha_vals}')
    print(f'M_0 = {cmass_M_0_vals}')
    print(f'sig_logm = {cmass_sig_logm_vals}')
    print('\n')
    print('WISE')
    print(f'M_min = {wise_M_min_vals}')
    print(f'M_1 = {wise_M_1_vals}')
    print(f'alpha = {wise_alpha_vals}')
    print(f'M_0 = {wise_M_0_vals}')
    print(f'sig_logm = {wise_sig_logm_vals}')
    print('\n')

    param_combos = product(
        cmass_M_min_vals,
        cmass_M_1_vals,
        cmass_alpha_vals,
        cmass_M_0_vals,
        cmass_sig_logm_vals,
        wise_M_min_vals,
        wise_M_1_vals,
        wise_alpha_vals,
        wise_M_0_vals,
        wise_sig_logm_vals
    )

    counter = 1
    best_loglike = -1e6
    best_cmass = []
    best_wise = []

    for combo in param_combos:
        loglike = likelihood_func(
            cmass_M_min = combo[0],
            cmass_M_1 = combo[1],
            cmass_alpha = combo[2],
            cmass_M_0 = combo[3],
            cmass_sig_logm = combo[4],
            wise_M_min = combo[5],
            wise_M_1 = combo[6],
            wise_alpha = combo[7],
            wise_M_0 = combo[8],
            wise_sig_logm = combo[9],
            R_ss = R_ss,
            R_cs = R_cs,
            R_sc = R_sc
        )

        if loglike > best_loglike:
            best_loglike = loglike 
            best_cmass = combo[:5]
            best_wise = combo[5:]

            print('\n')
            print(f'STEP {counter}')
            print(f'New best log-likelihood: {best_loglike}')
            print(f'CMASS parameters: {best_cmass}')
            print(f'WISE parameters: {best_wise}')
            print('\n')

        counter += 1

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
    output_file.write(f'M_min = {best_cmass[0]}\n')
    output_file.write(f'M_1 = {best_cmass[1]}\n')
    output_file.write(f'alpha = {best_cmass[2]}\n')
    output_file.write(f'M_0 = {best_cmass[3]}\n')
    output_file.write(f'sig_logm = {best_cmass[4]}\n')
    output_file.write('\n')
    output_file.write('WISE Parameters\n')
    output_file.write(f'M_min = {best_wise[0]}\n')
    output_file.write(f'M_1 = {best_wise[1]}\n')
    output_file.write(f'alpha = {best_wise[2]}\n')
    output_file.write(f'M_0 = {best_wise[3]}\n')
    output_file.write(f'sig_logm = {best_wise[4]}\n')
    output_file.close()

# ----------------------------------------------------------------------------------------------------------------------