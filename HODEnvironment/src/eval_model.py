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

def optimize_model(model_variations, loglike_func, method, packages_path, output='', debug=False):
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
    # Initialize model information dictionary
    info = {
        'params': model_variations.sampling_params_dict,
        'likelihood': {'my_cl_like': {'external': loglike_func}},
        'theory': {},
        'packages_path': packages_path,
        'sampler': {'minimize': 
            {'method': method,
            'ignore_prior': False,
            'override_scipy': {'method': 'Nelder-Mead'},
            'max_evals': 1e4,
            'confidence_for_unbounded': 0.9999995}},
    }

    if bool(output):
        info['output'] = output

    if debug:
        info['debug'] = True

    # Run optimizer
    return run(info)

# ----------------------------------------------------------------------------------------------------------------------
# Markov Chain Monte Carlo
# ----------------------------------------------------------------------------------------------------------------------

def mcmc(model_variations, loglike_func, packages_path, covmat_path, optim_path='', output='', debug=False):
    """
    Runs Markov Chain Monte Carlo (MCMC) chains on the CMASS-WISE HOD model using Cobaya's `run` function.

    Parameters
    ----------
    model_variations : ModelVariations
        Instance of the ModelVariations class that contains the HOD parameters to be sampled and the range over which
        they are sampled, and the HOD parameters to be held fixed.
    loglike_func : function
        Function that calculates the log-likelihood that the observed BOSS-CMASS and WISE data were produced by an
        HOD model with the BOSS-CMASS HOD model and WISE HOD model parameters.
    packages_path : str
        String representation of the path to the Cobaya `packages` directory.
    covmat_path : str
        String representation of the path to the covariance matrix to sample from.
    optim_path : str
        String representation of the path to the file giving the minimum found by the optimizer.
    output : str, optional
        String representation of the path to where the optimization results are saved.
    debug : bool, optional
        Determines whether Cobaya's debug mode should be used for more detailed console outputs.
    """
    # Initialize model information dictionary
    covmat_params = []
    for param in model_variations.sampling_params_dict.keys():
        try:
            #print('dict',model_variations.sampling_params_dict)
            #print('param',param)
            #print(model_variations.sampling_params_dict[param])
            #print(model_variations.sampling_params_dict[param]['ref'])
            if type(param) != dict:
                print('param',param)
                ref = model_variations.sampling_params_dict[param]['ref']
                covmat_params.append(param)
        except TypeError:
            continue
    info = {
        'params': model_variations.sampling_params_dict,
        'likelihood': {'my_cl_like': {'external': loglike_func, 'output_params': ['nbar_cmass','nbar_wise','bias_cmass','bias_wise','avg_mass_cmass',
        'avg_mass_wise','sat_frac_cmass','sat_frac_wise']}},
        'theory': {},
        'packages_path': packages_path,
        'sampler': {'mcmc': 
            {'learn_proposal': True,
            'oversample': True,
            'learn_proposal_Rminus1_max': 10,
            'proposal_scale': 1.0,
            'Rminus1_stop': 0.05,
            'covmat': np.loadtxt(covmat_path),
            'covmat_params': covmat_params,
            'burn_in': '0d',
            'max_tries': '100d'}},
    } 
    
    if optim_path != '':
        print('optim path',optim_path)
        optim_file = np.loadtxt(optim_path)
        try:
            info['params']['cmass_M_min']['ref'] = optim_file[2]
        except TypeError:
            pass
        print('info params cmass_M_1 ref',info['params']['cmass_M_1'], optim_file[3])
        try:
            info['params']['cmass_M_1']['ref'] = optim_file[3]
        except TypeError:
            pass
        try:
            info['params']['cmass_alpha']['ref'] = optim_file[4]
        except TypeError:
            pass
        try:
            if optim_file[5] >= 12:
                info['params']['cmass_M_0']['ref'] = optim_file[5]
            else:
                info['params']['cmass_M_0']['ref'] = 12.
        except TypeError:
            pass
        try:
            info['params']['cmass_sig_logm']['ref'] = optim_file[6]
        except TypeError:
            pass
        try:
            info['params']['wise_M_min']['ref'] = optim_file[7]
        except TypeError:
            pass
        try:
            info['params']['wise_M_1']['ref'] = optim_file[8]
        except TypeError:
            pass
        try:
            info['params']['wise_alpha']['ref'] = optim_file[9]
        except TypeError:
            pass
        try:
            info['params']['wise_M_0']['ref'] = optim_file[10]
        except TypeError:
            pass
        try:
            info['params']['wise_sig_logm']['ref'] = optim_file[11]
        except TypeError:
            pass

    if bool(output):
        info['output'] = output

    if debug:
        info['debug'] = True
    
    # Run chains
    return run(info)

# ----------------------------------------------------------------------------------------------------------------------
# Grid Search
# ----------------------------------------------------------------------------------------------------------------------

# Get grid search parameter ranges
def get_gridsearch_range(params, hod, param):
    """
    Determines the grid search parameter space for a given model parameter.

    Parameters
    ----------
    params : dict
        Dictionary of the CMASS-WISE HOD model parameters.
    hod : str
        The individual HOD model from which the parameter space will be determined.
    param : str
        The HOD model parameter whose parameter space will be determined.

    Returns
    -------
    params_vals : array-like
        Array representatino of the parameter's parameter space.
    """
    if params[hod][param]['sample']:
        param_vals = np.linspace(
            params[hod][param]['sample_min'],
            params[hod][param]['sample_max'],
            params[hod][param]['sample_div']
        )
    else:
        param_vals = np.array([params[hod][param]['val']])

    return param_vals

# Execute grid search
def gridsearch(params, loglike_func, output=''):
    """
    Finds the values of the model CMASS-WISE HOD model parameters that minimize the log-likelihood using a grid search.

    Parameters
    ----------
    params : dict
        Dictionary of the CMASS-WISE HOD model parameters.
    loglike_func : func
        Function that calculates the log-likelihood that the observed BOSS-CMASS and WISE data were produced by an
        HOD model with the BOSS-CMASS HOD model and WISE HOD model parameters.
    output : str, optional
        String representation of the path to where the optimization results are saved.

    Returns
    -------
    None
    """
    # Get parameter ranges
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

    # Print out parameter space values
    print('-'*80)
    print('Executing grid search over the following parameter space:')
    print('-'*80)
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
    print('-'*80)
    print('\n')

    # Get all possible parameter combinations
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

    # Initialize variables to keep track of search results
    counter = 1
    best_loglike = -1e6
    best_cmass = []
    best_wise = []

    # Execute grid search
    for combo in param_combos:
        loglike = loglike_func(
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

            print(f'STEP {counter}')
            print(f'New best log-likelihood: {loglike}')
            print(f'CMASS parameters: {best_cmass}')
            print(f'WISE parameters: {best_wise}')
            print('\n')

        counter += 1

    # Print results
    print('-'*80)
    print('GRID SEARCH COMPLETE')
    print(f'Best log-likelihood: {best_loglike}')
    print(f'CMASS parameters: {best_cmass}')
    print(f'WISE parameters: {best_wise}')
    print('-'*80)

    # Save results
    if bool(output):
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
