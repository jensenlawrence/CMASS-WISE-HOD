# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from get_model_info import get_model_params

# ----------------------------------------------------------------------------------------------------------------------
# ModelVariations Class
# ----------------------------------------------------------------------------------------------------------------------

class ModelVariations:
    """
    Generates the parameter space necessary for optimization and statistical analysis to be performed on the 
    CMASS-WISE HOD model.
    """
    # Initialize class
    def __init__(self, params_file):
        """
        Initializes the ModelVariations class.

        Parameters
        ----------
        params_file : str
            String representation of the path to the .json file containing the parameters for the BOSS-CMASS and WISE
            HOD models.

        Returns
        -------
        None
        """
        # Initialize parameter file attribute and get model parameters
        self.params_file = params_file 
        model_params = get_model_params(params_file)

        # Rename CMASS and WISE parameter keys
        cmass_keys = list(model_params['CMASS HOD'].keys())
        new_cmass_keys = ['cmass_' + key for key in cmass_keys]
        wise_keys = list(model_params['WISE HOD'].keys())
        new_wise_keys = ['wise_' + key for key in wise_keys]
        for i in range(len(cmass_keys)):
            model_params['CMASS HOD'][new_cmass_keys[i]] = model_params['CMASS HOD'][cmass_keys[i]]
            del model_params['CMASS HOD'][cmass_keys[i]]
            model_params['WISE HOD'][new_wise_keys[i]] = model_params['WISE HOD'][wise_keys[i]]
            del model_params['WISE HOD'][wise_keys[i]]

        # Initializelists for keys and values
        all_params_dict = {**model_params['CMASS HOD'], **model_params['WISE HOD'], **model_params['galaxy_corr']}
        del all_params_dict['cmass_central']
        del all_params_dict['wise_central']
        all_params_keys = list(all_params_dict.keys())
        all_params_vals = list(all_params_dict.values())
        
        # Determine fixed and sampled keys and values
        sample_params = []
        sample_range_mins = []
        sample_range_maxs = []
        sample_ranges = []
        sample_values = []

        fixed_params = []
        fixed_values = []

        for i in range(len(all_params_keys)):
            val = all_params_vals[i]
            if val['sample']:
                sample_params.append(all_params_keys[i])
                sample_range_mins.append(val['sample_min'])
                sample_range_maxs.append(val['sample_max'])
                sample_ranges.append(np.linspace(val['sample_min'], val['sample_max'], val['sample_div']))
                sample_values.append(val['val'])
            else:
                fixed_params.append(all_params_keys[i])
                fixed_values.append(val['val'])

        self.sample_params = sample_params
        self.sample_range_mins = sample_range_mins
        self.sample_range_maxs = sample_range_maxs 
        self.sample_ranges = sample_ranges 
        self.sample_values = sample_values

        self.fixed_params = fixed_params
        self.fixed_values = fixed_values

        # Create parameters dictionary
        params_dict = {}
        for i in range(len(self.fixed_params)):
            params_dict[self.fixed_params[i]] = self.fixed_values[i]
        for i in range(len(self.sample_params)):
            params_dict[self.sample_params[i]] = {'prior': {'min': self.sample_range_mins[i],
                                                            'max': self.sample_range_maxs[i]},
                                                  'ref': self.sample_values[i],
                                                  'latex': self.sample_params[i]}
        self.sampling_params_dict = params_dict

    # Printable representation of class instance
    def __str__(self):
        """
        Provides a printable representation of an instance of the ModelVariations class.
        """
        rep_str = '-'*80
        rep_str += '\nInstance of the ModelVariations class.'
        rep_str += '\n' + '-'*80
        for i in range(len(self.sample_params)):
            rep_str += f'\n- Samples {self.sample_params[i]}'
            rep_str += f' over [{self.sample_range_mins[i]}, {self.sample_range_maxs[i]}]'
            rep_str += f' with reference value {self.sample_values[i]}'
        rep_str += '\n' + '-'*80
        for i in range(len(self.fixed_params)):
            rep_str += f'\n- Holds {self.fixed_params[i]} fixed at {self.fixed_values[i]}'
        rep_str += '\n' + '-'*80

    # Equivalence of class instances
    def __eq__(self, other):
        """
        Compares an instance of the ModelVariations class to any other object.

        Parameters
        ----------
        other : any
            Any other object being compared against.

        Returns
        -------
        are_equal : bool
            True if other is an instance of the ModelVariations class with identical parameters, and False otherwise.
        """
        are_equal = isinstance(other, ModelVariations) and (self.sampled == other.sampled) and (self.fixed == other.fixed)
        return are_equal

# ----------------------------------------------------------------------------------------------------------------------