# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

from os import listdir
from os.path import isfile, join
import numpy as np
import matplotlib.pyplot as plt
from colour import Color
from getdist.mcsamples import MCSamples
from getdist import plots

# ----------------------------------------------------------------------------------------------------------------------
# Angular Cross-Correlation Plots
# ----------------------------------------------------------------------------------------------------------------------

# Plot CMASS autocorrelation function w(theta)
def cmass_autocorr_plot(cmass_wise_hod, sampled=[], plot_title='', output='', dpi=200):
    """
    Generates a plot of the observed and calculated CMASS angular autocorrelation functions.

    Parameters
    ----------
    cmass_wise_hod : CMASS_WISE_HOD
        The instance of the CMASS_WISE_HOD class whose observed and calculated CMASS autocorrelations will
        be plotted.
    sampled : array-like, optional
        Array-like object of the form [sampled_param, sampled_range].
        sampled_param : str
            String representation of one of the model parameters to be sampled over during graphing.
        sampled_range : array-like
            Array containing the range of values over which sampled_param will be sampled.
    plot_title : str, optional
        String representation of the plot title.
    output : str, optional
        String representation of the path and file name the plot will be saved as.
    dpi : int, optional
        The dots-per-inch (resolution) of the graph.

    Returns
    -------
    None
    """
    # Plot while varying one parameter
    if bool(sampled): 
        sampled_param = sampled[0]
        sampled_range = sampled[1]

        min_colour = Color('#00FFFA')
        max_colour = Color('#4B0083')
        colour_range = list(min_colour.range_to(max_colour, len(sampled_range)))

        counter = 0
        for sampled_value in sampled_range:
            cmass_auto = cmass_wise_hod.cmass_auto
            cmass_auto.update(hod_params={sampled_param: sampled_value})
            plt.plot(cmass_wise_hod.thetas[3:], cmass_wise_hod.corr_cmass_auto(), str(colour_range[counter]),
                        label=f'{sampled_param} = {sampled_value}')
            counter += 1

    # Plot holding all parameters fixed
    else:
        plt.plot(cmass_wise_hod.thetas[3:], cmass_wise_hod.corr_cmass_auto(), color='dodgerblue', label='Model')
        
    plt.errorbar(cmass_wise_hod.thetas[3:], cmass_wise_hod.data[:7,1], np.sqrt(np.diag(cmass_wise_hod.covariance[:7,:7])),
                 color='springgreen', label='CMASS Autocorrelation Data')

    # Plot formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta$', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel(r'$w(\theta)$', fontsize=12)
    plt.yticks(fontsize=10)
    plt.legend(loc='best', fontsize=10)

    if bool(plot_title):
        plt.title(plot_title, fontsize=9)

    if bool(output):
        plt.savefig(output, dpi=dpi)

    else:
        plt.savefig('cmass_autocorr_plot.png', dpi=dpi)

    plt.close()

# Plot CMASS-WISE cross-correlation function w(theta)
def crosscorr_plot(cmass_wise_hod, sampled=[], plot_title='', output='', dpi=200):
    """
    Generates a plot of the observed and calculated CMASS-WISE angular cross-correlation functions.

    Parameters
    ----------
    cmass_wise_hod : CMASS_WISE_HOD
        The instance of the CMASS_WISE_HOD class whose observed and calculated CMASS-WISE cross-correlations will
        be plotted.
    sampled : array-like, optional
        Array-like object of the form [sampled_param, sampled_range].
        sampled_param : str
            String representation of one of the model parameters to be sampled over during graphing.
        sampled_range : array-like
            Array containing the range of values over which sampled_param will be sampled.
    plot_title : str, optional
        String representation of the plot title.
    output : str, optional
        String representation of the path and file name the plot will be saved as.
    dpi : int, optional
        The dots-per-inch (resolution) of the graph.

    Returns
    -------
    None
    """
    # Plot while varying one parameter
    if bool(sampled): 
        sampled_param = sampled[0]
        sampled_range = sampled[1]

        min_colour = Color('#00FFFA')
        max_colour = Color('#4B0083')
        colour_range = list(min_colour.range_to(max_colour, len(sampled_range)))

        counter = 0
        for sampled_value in sampled_range:
            cross = cmass_wise_hod.cross
            cross.halo_model_2.update(hod_params={sampled_param: sampled_value})
            plt.plot(cmass_wise_hod.thetas, cmass_wise_hod.corr_cross(), str(colour_range[counter]),
                        label=f'{sampled_param} = {sampled_value}')
            counter += 1

    # Plot holding all parameters fixed
    else:
        plt.plot(cmass_wise_hod.thetas, cmass_wise_hod.corr_cross(), color='dodgerblue', label='Model')
        
    plt.errorbar(cmass_wise_hod.thetas, cmass_wise_hod.data[7:,1], np.sqrt(np.diag(cmass_wise_hod.covariance[7:,7:])),
                 color='springgreen', label='Cross-correlation Data')

    # Plot formatting
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$\theta$', fontsize=12)
    plt.xticks(fontsize=10)
    plt.ylabel(r'$w(\theta)$', fontsize=12)
    plt.yticks(fontsize=10)
    plt.legend(loc='best', fontsize=10)

    if bool(plot_title):
        plt.title(plot_title, fontsize=9)

    if bool(output):
        plt.savefig(output, dpi=dpi)

    else:
        plt.savefig('crosscorr_plot.png', dpi=dpi)

    plt.close()

# Generate titles for w(theta) plots
def get_corr_title(params, loglike_func):
    """
    Generates titles for the auto-correlation and cross-correlation plots.

    Parameters
    ----------
    params : dict
        Dictionary of the CMASS-WISE HOD model parameters.
    loglike_func : function
        Function that calculates the log-likelihood that the observed BOSS-CMASS and WISE data were produced by an
        HOD model with the BOSS-CMASS HOD model and WISE HOD model parameters.

    Returns
    -------
    title : str
        Correlation plot title.
    """
    # Get CMASS title components
    cmass_s1 = r'$M_{\min} = $' + f'{params["CMASS HOD"]["M_min"]["val"]}'
    cmass_s2 = r'$M_{1} = $' + f'{params["CMASS HOD"]["M_1"]["val"]}'
    cmass_s3 = r'$\alpha = $' + f'{params["CMASS HOD"]["alpha"]["val"]}'
    cmass_s4 = r'$M_{0} = $' + f'{params["CMASS HOD"]["M_0"]["val"]}'
    cmass_s5 = r'$\sigma_{\log{M}} = $' + f'{params["CMASS HOD"]["sig_logm"]["val"]}'
    cmass_s6 = f'central = {params["CMASS HOD"]["central"]["val"]}'
    cmass_title = f'CMASS : {cmass_s1}, {cmass_s2}, {cmass_s3}, {cmass_s4}, {cmass_s5}, {cmass_s6}\n'

    # Get WISE title components
    wise_s1 = r'$M_{\min} = $' + f'{params["WISE HOD"]["M_min"]["val"]}'
    wise_s2 = r'$M_{1} = $' + f'{params["WISE HOD"]["M_1"]["val"]}'
    wise_s3 = r'$\alpha = $' + f'{params["WISE HOD"]["alpha"]["val"]}'
    wise_s4 = r'$M_{0} = $' + f'{params["WISE HOD"]["M_0"]["val"]}'
    wise_s5 = r'$\sigma_{\log{M}} = $' + f'{params["WISE HOD"]["sig_logm"]["val"]}'
    wise_s6 = f'central = {params["WISE HOD"]["central"]["val"]}'
    wise_title = f'WISE : {wise_s1}, {wise_s2}, {wise_s3}, {wise_s4}, {wise_s5}, {wise_s6}\n'

    # Get R title components
    R_s1 = r'$R_{ss} = $' + f'{params["galaxy_corr"]["R_ss"]["val"]}'
    R_s2 = r'$R_{cs} = $' + f'{params["galaxy_corr"]["R_cs"]["val"]}'
    R_s3 = r'$R_{sc} = $' + f'{params["galaxy_corr"]["R_sc"]["val"]}'

    # Calculate log-likliehood
    likelihood = loglike_func(
        cmass_M_min = params["CMASS HOD"]["M_min"]["val"],
        cmass_M_1 = params["CMASS HOD"]["M_1"]["val"],
        cmass_alpha = params["CMASS HOD"]["alpha"]["val"],
        cmass_M_0 = params["CMASS HOD"]["M_0"]["val"],
        cmass_sig_logm = params["CMASS HOD"]["sig_logm"]["val"],
        wise_M_min = params["WISE HOD"]["M_min"]["val"],
        wise_M_1 = params["WISE HOD"]["M_1"]["val"],
        wise_alpha = params["WISE HOD"]["alpha"]["val"],
        wise_M_0 = params["WISE HOD"]["M_0"]["val"],
        wise_sig_logm = params["WISE HOD"]["sig_logm"]["val"],
        R_ss = params["galaxy_corr"]["R_ss"]["val"],
        R_cs = params["galaxy_corr"]["R_cs"]["val"],
        R_sc = params["galaxy_corr"]["R_sc"]["val"]
    )
    R_title = f'{R_s1}, {R_s2}, {R_s3}'
    likelihood_title = f'Log-likelihood = {likelihood}'
    R_title += f' | {likelihood_title}'

    # Get complete title
    title = cmass_title + wise_title + R_title

    return title

# ----------------------------------------------------------------------------------------------------------------------
# MCMC Posterior Plot
# ----------------------------------------------------------------------------------------------------------------------

# Function for plotting posterior distributions from MCMC chains
def posterior_plot(samples_path, names, labels, output=''):
    """
    Generates a plot of the posterior distribution of the CMASS-WISE HOD model parameters determined by an MCMC chain.

    Parameters
    ----------
    samples_path : str
        String representation of the path to the MCMC chain results.
    names : array-like
        Array of strings containing the names of each variable in the posterior distribution. Order should reflect the
        order of the columns in the posterior distribution file.
    labels : array-like
        Array of strings containing the LaTex representations of each variable in the posterior distribution. Order
        should reflect the order of the columns in the posterior distribution file.
    output : Str
        String representation of the path and file name the plot will be saved as.

    Returns
    -------
    None
    """
    # Get all files in target directory
    samples_dir = '/'.join(samples_path.split('/')[:-1]) + '/'
    files_only = [f for f in listdir(samples_dir) if isfile(join(samples_dir, f))]

    # Determine which files in target directory are desired MCMC chain results files
    samples_name = samples_path.split('/')[-1]
    sample_files = []
    for file in files_only:
        if (samples_name in file) and ('.txt' in file):
            sample_files.append(file)

    # Load data from MCMC chains files
    sample_data = []
    for i in range(len(sample_files)):
        load_sample = np.loadtxt(samples_dir + sample_files[i])
        sample_vals = []
        for j in range(len(load_sample)):
            step_vals = []
            for k in range(len(names)):
                step_vals.append(load_sample[j][k + 2])
            sample_vals.append(step_vals)
        
        # Remove first 30% of data to remove bad values due to burn-in
        sample_vals = sample_vals[int(np.ceil(0.3*len(sample_vals))):]
        sample_data.append(np.array(sample_vals))

    # Load MC samples from data
    samples = []
    for i in range(len(sample_files)):
        sample = MCSamples(samples = sample_data[i], names = names, labels = labels, label = f'Chain {i + 1}')
        samples.append(sample)

    # Plot posterior distribution
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, filled=True)

    if bool(output):
        save_path = '/'.join(output.split('/')[:-1])
        save_name = output.split('/')[-1]
        g.export(save_name, save_path)

    else:
        g.export(samples_name + '.png', samples_dir)

# ----------------------------------------------------------------------------------------------------------------------