# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import json
from astropy.cosmology import Planck15

# ----------------------------------------------------------------------------------------------------------------------
# Model Parameter Extraction and Dictionary Creation
# ----------------------------------------------------------------------------------------------------------------------

# Getting model parameters
def get_model_params(params_file):
    """
    get_model_params : str -> dict
        Returns a combined dictionary of the BOSS-CMASS and WISE HOD model parameters provided in params_file.

    params_file : str
        String representation of the path to the .json file containing the parameters for the BOSS-CMASS and
        WISE HOD models.
    """
    with open(params_file) as f:
        model_params = json.load(f)
    f.close()

    if model_params['halo_params']['cosmo_model'] == 'Planck15':
        model_params['halo_params']['cosmo_model'] = Planck15

    return model_params

# Generating model dictionaries
def get_model_dicts(params_file):
    """
    get_model_dicts : str -> dict, dict
        Returns a dictionary of the BOSS-CMASS HOD model parameters and a dictionary of the WISE HOD model
        parameters provided in params_file.

    params_file : str
        String representation of the path to the .json file containing the parameters for the BOSS-CMASS and
        WISE HOD models.
    """
    model_params = get_model_params(params_file)
    halo_params = model_params['halo_params']
    del halo_params['zmin']
    del halo_params['zmax']

    cmass_model = halo_params.copy()
    cmass_model['hod_params'] = {
        'M_min': model_params['CMASS HOD']['M_min']['val'],
        'M_1': model_params['CMASS HOD']['M_1']['val'],
        'alpha': model_params['CMASS HOD']['alpha']['val'],
        'M_0': model_params['CMASS HOD']['M_0']['val'],
        'sig_logm': model_params['CMASS HOD']['sig_logm']['val'],
        'central': model_params['CMASS HOD']['central']['val']
    }

    wise_model = halo_params.copy()
    wise_model['hod_params'] = {
        'M_min': model_params['WISE HOD']['M_min']['val'],
        'M_1': model_params['WISE HOD']['M_1']['val'],
        'alpha': model_params['WISE HOD']['alpha']['val'],
        'M_0': model_params['WISE HOD']['M_0']['val'],
        'sig_logm': model_params['WISE HOD']['sig_logm']['val'],
        'central': model_params['WISE HOD']['central']['val']
    }

    return cmass_model, wise_model

# ----------------------------------------------------------------------------------------------------------------------