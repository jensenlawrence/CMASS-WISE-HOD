# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------------------------------------------------
# Angular Cross-Correlation Plot
# ----------------------------------------------------------------------------------------------------------------------

# Function for plotting CMASS-WISE w(theta)
def wtheta_plot(cross_hod, sampled=[], show_errorbars=True, plot_title='', save_as='', dpi=400):
    """
    Docstring goes here
    """
    if bool(sampled): 
        sampled_param = sampled[0]
        sampled_range = sampled[1]

        min_colour = Color('#00FFFA')
        max_colour = Color('#4B0083')
        colour_range = list(min_colour.range_to(max_colour, len(sampled_range)))

        if show_theory:
            counter = 0
            for sampled_value in sampled_range:
                cross = cross_hod.cross
                cross.halo_model_2.update(hod_params={sampled_param: sampled_value})
                plt.plot(cross_hod.thetas, cross_hod.corr_cross(), str(colour_range[counter]),
                         label=f'{sampled_param} = {sampled_value}')
                counter += 1

    else:
        plt.plot(cross_hod.thetas, cross_hod.corr_cross(), color='dodgerblue', label='Theory')
        
    plt.errorbar(cross_hod.thetas, cross_hod.data[:,2], np.sqrt(np.diag(cross_hod.covariance[10:,10:])),
                 color='springgreen', label='Data')

    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta$', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel(r'$w(\theta)$', fontsize=12)
    plt.yticks(fontsize=10)
    plt.legend(loc='best', fontsize=10)

    if bool(plot_title):
        plt.title(plot_title, fontsize=9)

    if bool(save_as):
        plt.savefig(save_as, dpi=dpi)

    else:
        plt.savefig('wtheta_plot.png', dpi=dpi)

# ----------------------------------------------------------------------------------------------------------------------
# MCMC Posterior Plot
# ----------------------------------------------------------------------------------------------------------------------

# Function for plotting posterior distributions from MCMC chains
def posterior_plot(samples_path, names, axis_labels, legend_labels, save_as=''):
    # needs to be implemented
    pass

# ----------------------------------------------------------------------------------------------------------------------