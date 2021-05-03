# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

from cobaya.model import get_model
from cobaya.run import run

# ----------------------------------------------------------------------------------------------------------------------
# Model Optimization
# ----------------------------------------------------------------------------------------------------------------------

# Optimizes varied parameters of specified model
def cobaya_optimize(model_variations, likelihood_func, method, packages_path, output='', debug=False):
    """
    Docstring goes here
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
    Docstring goes here
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