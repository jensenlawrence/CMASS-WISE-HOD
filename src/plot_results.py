# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from getdist.mcsamples import MCSamples, loadMCSamples
from getdist import gaussian_mixtures, plots

# ----------------------------------------------------------------------------------------------------------------------
# Angular Cross-Correlation Plots
# ----------------------------------------------------------------------------------------------------------------------

# Function for plotting CMASS autocorrelated w(theta)
def cmass_autocorr_plot(cross_hod, sampled=[], plot_title='', save_as='', dpi=400):
    """
    cmass_autocorr_plot : CrossHOD, array-like, str, str, int -> Plot
        Produces a figure showing the observed and calculated angular autocorrelation functions for
        the CMASS autocorrelation.

    cross_hod : CrossHOD
        The instance of a CrossHOD class containing the relevant galaxy distribution data and HOD models required
        for constructing the angular correlation vs. theta plot.

    sampled : Array-like
        Array-like object of the form [sampled_param, sampled_range]. Optional argument.
            sampled_param : Str
                String representation of one of the parameters to be sampled over during graphing.
            sampled_range : Array-like
                Array-like object which contains the values over which the related sampled_param will be sampled.

    plot_title : Str
        String representation of the title of the plot that will be produced.
        Default value is ''.

    save_as : Str
        String representation of the file name the plot will be saved under.
        Default value is ''.

    dpi : Int
        Values specifying the dpi (measure of resolution) of the plot that will be produced.
        Default value is 400.
    """
    # Plotting while varying one parameter
    if bool(sampled): 
        sampled_param = sampled[0]
        sampled_range = sampled[1]

        min_colour = Color('#00FFFA')
        max_colour = Color('#4B0083')
        colour_range = list(min_colour.range_to(max_colour, len(sampled_range)))

        counter = 0
        for sampled_value in sampled_range:
            cmass_auto = cross_hod.cmass_auto
            cmass_auto.update(hod_params={sampled_param: sampled_value})
            plt.plot(cross_hod.thetas[3:], cross_hod.corr_cmass_auto(), str(colour_range[counter]),
                        label=f'{sampled_param} = {sampled_value}')
            counter += 1

    # Plotting holding all parameters fixed
    else:
        plt.plot(cross_hod.thetas[3:], cross_hod.corr_cmass_auto(), color='dodgerblue', label='Model')
        
    plt.errorbar(cross_hod.thetas[3:], cross_hod.data[:7,1], np.sqrt(np.diag(cross_hod.covariance[:7,:7])),
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

    if bool(save_as):
        plt.savefig(save_as, dpi=dpi)

    else:
        plt.savefig('cmass_autocorr_plot.png', dpi=dpi)

    plt.close()

# Function for plotting CMASS-WISE cross-correlated w(theta)
def crosscorr_plot(cross_hod, sampled=[], plot_title='', save_as='', dpi=400):
    """
    crosscorr_plot : CrossHOD, array-like, str, str, int -> Plot
        Produces a figure showing the observed and calculated angular cross-correlation functions for
        the CMASS-WISE cross-correlation.

    cross_hod : CrossHOD
        The instance of a CrossHOD class containing the relevant galaxy distribution data and HOD models required
        for constructing the angular correlation vs. theta plot.

    sampled : Array-like
        Array-like object of the form [sampled_param, sampled_range]. Optional argument.
            sampled_param : Str
                String representation of one of the parameters to be sampled over during graphing.
            sampled_range : Array-like
                Array-like object which contains the values over which the related sampled_param will be sampled.

    plot_title : Str
        String representation of the title of the plot that will be produced.
        Default value is ''.

    save_as : Str
        String representation of the file name the plot will be saved under.
        Default value is ''.

    dpi : Int
        Values specifying the dpi (measure of resolution) of the plot that will be produced.
        Default value is 400.
    """
    # Plotting while varying one parameter
    if bool(sampled): 
        sampled_param = sampled[0]
        sampled_range = sampled[1]

        min_colour = Color('#00FFFA')
        max_colour = Color('#4B0083')
        colour_range = list(min_colour.range_to(max_colour, len(sampled_range)))

        counter = 0
        for sampled_value in sampled_range:
            cross = cross_hod.cross
            cross.halo_model_2.update(hod_params={sampled_param: sampled_value})
            plt.plot(cross_hod.thetas, cross_hod.corr_cross(), str(colour_range[counter]),
                        label=f'{sampled_param} = {sampled_value}')
            counter += 1

    # Plotting holding all parameters fixed
    else:
        plt.plot(cross_hod.thetas, cross_hod.corr_cross(), color='dodgerblue', label='Model')
        
    plt.errorbar(cross_hod.thetas, cross_hod.data[7:,1], np.sqrt(np.diag(cross_hod.covariance[7:,7:])),
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

    if bool(save_as):
        plt.savefig(save_as, dpi=dpi)

    else:
        plt.savefig('crosscorr_plot.png', dpi=dpi)

    plt.close()

# Title generating function
def get_corr_title(params, likelihood_func):
    """
    Docstring goes here
    """
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
    likelihood = likelihood_func(
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

    title = cmass_title + wise_title + R_title

    return title

# ----------------------------------------------------------------------------------------------------------------------
# MCMC Posterior Plot
# ----------------------------------------------------------------------------------------------------------------------

# Function for plotting posterior distributions from MCMC chains
def posterior_plot(samples_path, names, labels, save_as=''):
    """
    posterior_plot : str, array-like, array-like, str -> Plot
        Returns a plot of the posterior distribution determined from the MCMC chain results stored at samples_path.

    samples_path : str
        String representation of the path to the MCMC chain results.

    names : array-like
        Array of strings containing the names of each variable in the posterior distribution. Order should reflect the
        order of the columns in the posterior distribution file.

    labels : array-like
        Array of strings containing the LaTex representations of each variable in the posterior distribution. Order
        should reflect the order of the columns in the posterior distribution file.

    save_as : Str
        String representation of the file name the plot will be saved under.
        Default value is ''.
    """
    # Getting all files in target directory
    samples_dir = '/'.join(samples_path.split('/')[:-1]) + '/'
    files_only = [f for f in listdir(samples_dir) if isfile(join(samples_dir, f))]

    # Determining which files in target directory are desired MCMC chain results files
    samples_name = samples_path.split('/')[-1]
    sample_files = []
    for file in files_only:
        if (samples_name in file) and ('.txt' in file):
            sample_files.append(file)

    # Loading data from MCMC chains files
    sample_data = []
    for i in range(len(sample_files)):
        load_sample = np.loadtxt(samples_dir + sample_files[i])
        sample_vals = []
        for j in range(len(load_sample)):
            step_vals = []
            for k in range(len(names)):
                step_vals.append(load_sample[j][k + 2])
            sample_vals.append(step_vals)
        
        # Removing first 30% of data to remove bad values due to burn-in
        sample_vals = sample_vals[int(np.ceil(0.3*len(sample_vals))):]
        sample_data.append(np.array(sample_vals))

    # Loading MC samples from data
    samples = []
    for i in range(len(sample_files)):
        sample = MCSamples(samples = sample_data[i], names = names, labels = labels, label = f'Chain {i + 1}')
        samples.append(sample)

    # Plotting posterior distribution
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, filled=True)

    if bool(save_as):
        save_path = '/'.join(save_as.split('/')[:-1])
        save_name = save_as.split('/')[-1]
        g.export(save_name, save_path)

    else:
        g.export(samples_name + '.png', samples_dir)

# ----------------------------------------------------------------------------------------------------------------------