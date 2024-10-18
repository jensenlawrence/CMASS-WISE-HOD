# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# Basic imports
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

# Cosmology imports
sys.path.append('/home/akrolews/unwise_hod/HODEnvironment/src')
from cross_correlations import HODCross
from astropy.cosmology import Planck15

# Custom imports
from get_model_info import get_model_params
from cmass_wise_hod_beyond_limber import CMASS_WISE_HOD
from model_variations import ModelVariations
from eval_model import optimize_model, mcmc, gridsearch
#from plot_results import cmass_autocorr_plot, crosscorr_plot, get_corr_title, posterior_plot
#import halofit_test
import time
from mpi4py import MPI

rank = MPI.COMM_WORLD.Get_rank()



# ----------------------------------------------------------------------------------------------------------------------
# Program Execution
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="""HOD model for the cross-correlation of BOSS-CMASS and WISE
                                                    galaxies at a redshift of z ~ 0.5.""")
    parser.add_argument('-a', '--action', type=str, metavar='action',
                        help="""Function executed by the program. Options are: optimize, mcmc, gridsearch, corr_plots,
                                posterior_plot.""")
    parser.add_argument('--zmin', type=str, metavar='zmin',
                        help="""Minimum redshift of the spec bin""")
    parser.add_argument('--zmax', type=str, metavar='zmin',
                        help="""Maximum redshift of the spec bin""")
    parser.add_argument('--version', type=str, metavar='version')
    parser.add_argument('--json', type=str, metavar='json')
    

    args = parser.parse_args()

    # Verify argument is valid
    assert args.action in ('optimize', 'mcmc', 'gridsearch', 'corr_plots',
                           'posterior_plot', 'test', 'test_vs_halofit'), 'Invalid action chosen.'

    # ----------------------------------------------------------------------------------------------------------------------
    # Packages, Data, and Parameter Paths
    # ----------------------------------------------------------------------------------------------------------------------

    # packages_path = '/home/jptlawre/packages'
    # cmass_redshift_file = '/home/jptlawre/projects/rrg-wperciva/jptlawre/data/dr12cmassN-r1-v2-flag-wted-convolved-0.45-0.5.txt'
    # wise_redshift_file = '/home/jptlawre/projects/rrg-wperciva/jptlawre/data/blue.txt'
    # data_file = '/home/jptlawre/projects/rrg-wperciva/jptlawre/data/combined_data.txt'
    # covariance_file = '/home/jptlawre/projects/rrg-wperciva/jptlawre/data/combined_cov.txt'
    # params_file = '/home/jptlawre/scratch/wca/cmass_wise_params_cross.json'
    # params = get_model_params(params_file)

    zmin = float(args.zmin)
    zmax = float(args.zmax)
    version = args.version
    params_file = args.json

    packages_path = '/home/jptlawre/packages'
    cmass_redshift_file = 'dndz/dr12cmassN-r1-v2-flag-wted-convolved-%.2f-%.2f.txt' % (zmin, zmax)
    wise_redshift_file = 'dndz/blue.txt'

    data_file = 'data/data/combined_data_wise_wts_z%.2f_%.2f.txt' % (zmin, zmax)
    covariance_file = 'data/cov/combined_data_wise_wts_cov_z%.2f_%.2f_percival.txt' % (zmin, zmax)
    # params_file = '/home/jptlawre/scratch/wca/json/cmass_wise_params_cross_%.2f_%.2f_v2.json' % (zmin, zmax)
    # params_file = f'/home/jptlawre/scratch/wca/json/{args.json}'

    magbias1 = 'data/magbias/unwise_DR12_cmass_zmin_%.2f_zmax_%.2f_magbias_wise_mu_cross_spec_g.txt' % (zmin, zmax)
    magbias2 = 'data/magbias/unwise_DR12_cmass_zmin_%.2f_zmax_%.2f_magbias_wise_g_cross_spec_mu.txt' % (zmin, zmax)
    magbias3 = 'data/magbias/unwise_DR12_cmass_zmin_%.2f_zmax_%.2f_magbias_mumu.txt' % (zmin, zmax)

    cmass_bias_file = np.loadtxt('cmass_auto_bias_NEW.txt')
    fiducial_cmass_bias = cmass_bias_file[:,2][(cmass_bias_file[:,0] < 0.5 * (zmin + zmax)) & (cmass_bias_file[:,1] > 0.5 * (zmin + zmax))]

    params = get_model_params(params_file)
    
    derived_name = f'derived/mcmc{version}-{zmin:.2f}-{zmax:.2f}.{rank:d}.txt'
    
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

    # Instance of CMASS_WISE_HOD
    hod = CMASS_WISE_HOD(
        cmass_redshift_file = cmass_redshift_file,
        wise_redshift_file = wise_redshift_file,
        data_file = data_file,
        covariance_file = covariance_file,
        params_file = params_file,
        magbias1 = magbias1,
        magbias2 = magbias2,
        magbias3 = magbias3,
        fiducial_cmass_bias = fiducial_cmass_bias,
        cross_hod_model = VariableCorr,
        diag_covariance = False, 
        exclusion_model = 'NgMatched',
        exclusion_params = None,
        min_bin = params['min_bin'],
        derived_name = derived_name
    )
    
    # nbar-weighted log-likelihood
    def nbar_loglike(cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, wise_M_min, wise_M_1,
                        wise_alpha, wise_M_0, wise_sig_logm, R_ss, R_cs, R_sc):
        """
        Calculates the individual components of the nbar-weighted log-likelihood that the observed BOSS-CMASS and
        WISE data were produced by an HOD model with the BOSS-CMASS HOD model and WISE HOD model parameters.

        Parameters
        ----------
        See `loglike_components`.

        Returns
        -------
        loglike_dict : dict
            Dictionary containing the autocorrelation-only, cross-correlation-only, and total log-likelihood values,
            as well as the data-based and model-based galaxy number densities, and their associated number density
            corrections.
        """
        sig_nbar = 0.1
        
        cmass_nbar_model, wise_nbar_model, total_loglike = hod.loglike(cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm,
                               wise_M_min, wise_M_1, wise_alpha, wise_M_0, wise_sig_logm, R_ss,
                               R_cs, R_sc)


        # Get CMASS number densities and calculate CMASS number density correction
        #if (hod.corr_flag == "combined"):
        #    cmass_nbar_data = hod.model_params['nbar']['CMASS']
        #    cmass_nbar_model = hod.cross.halo_model_1.mean_tracer_den
        #    cmass_nbar_correction = -0.5 * ((cmass_nbar_data - cmass_nbar_model)/(sig_nbar * cmass_nbar_data))**2
        if (hod.corr_flag == "cmass_only") or (hod.corr_flag == 'combined') or (hod.corr_flag == 'wise_incompleteness'):
            cmass_nbar_data = hod.model_params['nbar']['CMASS']
            cmass_nbar_err = hod.model_params['nbar_err']['CMASS']
            wise_nbar_err = hod.model_params['nbar_err']['WISE']
            print('cmass_nbar_data',cmass_nbar_data)
            print('cmass_nbar_model',cmass_nbar_model)
            cmass_nbar_correction = -0.5 * ((cmass_nbar_data - cmass_nbar_model)/(cmass_nbar_err))**2

        # Get WISE number densities and calculate WISE number density correction
        if (hod.corr_flag == "cross_only") or (hod.corr_flag == "combined"):
            wise_nbar_data = hod.model_params['nbar']['WISE']
            wise_nbar_err = hod.model_params['nbar_err']['WISE']
            print('wise_nbar_data',wise_nbar_data)
            print('wise_nbar_model',wise_nbar_model)
            wise_nbar_correction = -0.5 * ((wise_nbar_data - wise_nbar_model)/(wise_nbar_err))**2
            #wise_nbar_correction = -100./(1 + np.exp(-(wise_nbar_data  - wise_nbar_model)/(sig_nbar * wise_nbar_data)))
            #if wise_nbar_model > wise_nbar_data + sig_nbar * wise_nbar_data:
            #    wise_nbar_correction = -np.inf
            #else:
            #    wise_nbar_correction = 0

        # Get unweighted log-likelihood components
        if hod.corr_flag == 'wise_incompleteness':
            wise_nbar_data = hod.model_params['nbar']['WISE']
            wise_nbar_err = hod.model_params['nbar_err']['WISE']
            print('wise_nbar_data',wise_nbar_data)
            print('wise_nbar_model',wise_nbar_model)
            wise_nbar_correction = -0.5 * ((wise_nbar_data - wise_nbar_model)/(wise_nbar_err))**2
            if wise_nbar_data < wise_nbar_model:
                wise_nbar_correction = 0


        if hod.corr_flag == "cmass_only":
            #print(f'total_loglike = {total_loglike}')
            #print(f'cmass_nbar_correction = {cmass_nbar_correction}')
            corrected_loglike = total_loglike + cmass_nbar_correction

        if hod.corr_flag == "cross_only":
            corrected_loglike = total_loglike + wise_nbar_correction

        if (hod.corr_flag == "combined") or (hod.corr_flag == 'wise_incompleteness'):
            #print(f'total_loglike = {total_loglike}')
            print(f'cmass_nbar_correction = {cmass_nbar_correction}')
            print(f'wise_nbar_correction = {wise_nbar_correction}')

            corrected_loglike = total_loglike + cmass_nbar_correction + wise_nbar_correction
        print(f'corrected_loglike = {corrected_loglike}')
        
        derived_file = open(hod.derived_name, 'a')
        derived_file.write('%.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5f %.5e %.5e %.5f %.5f %.5f %.5f %.5f %.5f\n'
        % (corrected_loglike, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, wise_M_min, wise_M_1,
            wise_alpha, wise_M_0, wise_sig_logm, cmass_nbar_model, wise_nbar_model,
            hod.cross.b1, hod.cross.b2, hod.cross.m1, hod.cross.m2, hod.cross.halo_model_1.satellite_fraction,
            hod.cross.halo_model_2.satellite_fraction))
        derived_file.close()
        
        #print('dir HM1',dir(hod.cross.halo_model_1))
        #print('HM1 params',hod.cross._halo_model_1_params)
        #print('HM2 params',hod.cross._halo_model_2_params)
        #print('central HM2',hod.cross.halo_model_2.central)
        #print(5/0)

        return (corrected_loglike, {'nbar_cmass': cmass_nbar_model, 'nbar_wise': wise_nbar_model,
        'bias_cmass': hod.cross.b1, 'bias_wise': hod.cross.b2,
        'avg_mass_cmass': hod.cross.m1, 'avg_mass_wise': hod.cross.m2, 
        'sat_frac_cmass': hod.cross.halo_model_1.satellite_fraction, 
        'sat_frac_wise': hod.cross.halo_model_2.satellite_fraction})


    # Instance of ModelVariations
    cmass_wise_variations = ModelVariations(params_file)


    # Optimizer action
    if args.action == 'optimize':
        
        # Set optimizer output
        output = f'/home/jptlawre/scratch/wca/optim2/optim{version}-{zmin:.2f}-{zmax:.2f}'
        # output = '/home/jptlawre/scratch/wca/optim2/optim7-%.2f-%.2f' % (zmin, zmax) 
        if output == '':
            output = input('Enter optimizer output path: ')
        
        # Run optimizer
        optimize_model(
            model_variations = cmass_wise_variations,
            loglike_func = hod.nbar_loglike,
            method = 'scipy',
            packages_path = packages_path,
            output = output,
            debug = True
        )

    # MCMC action
    elif args.action == 'mcmc':

        # Set MCMC output
        output = f'results/mcmc{version}-{zmin:.2f}-{zmax:.2f}'
        # output = 'results/mcmc15-%.2f-%.2f' % (zmin, zmax)
        if output == '':
            output = input('Enter MCMC output path: ')

        # Run MCMC chains
        mcmc(
            model_variations = cmass_wise_variations,
            loglike_func = nbar_loglike,
            packages_path = packages_path,
            output = output,
            covmat_path = params['covmat_path'],
            #optim_path = '/home/jptlawre/scratch/wca/optim2/optim7-%.2f-%.2f.minimum.txt' % (zmin, zmax),
            debug = True
        )

    # Grid search action
    elif args.action == 'gridsearch':

        # Set grid search output
        output = 'results/gridsearch/grid1'
        if output == '':
            output = input('Enter grid search output path: ')

        # Run grid search
        gridsearch(
            params = params,
            loglike_func = hod.nbar_components,
            output = output
        )

    # Correlation plots action
    elif args.action == 'corr_plots':

        # Set autocorr plot output
        auto_output = ''
        if auto_output == '':
            auto_output = input('Enter autocorrelation plot output path: ')

        # Set cross-corr plot output
        cross_output = ''
        if cross_output == '':
            cross_output = input('Enter cross-correlation plot output path: ')

        # Generate correlation plots
        title = get_corr_title(params, hod.nbar_loglike)

        cmass_autocorr_plot(
            cmass_wise_hod = hod,
            sampled = [],
            plot_title = title,
            output = auto_output,
            dpi = 200
        )  

        crosscorr_plot(
            cmass_wise_hod = hod,
            sampled = [],
            plot_title = title,
            output = cross_output,
            dpi = 200
        )

    # Posterior plot action
    elif args.action == 'posterior_plot':

        # Set samples, names, and labels
        samples_path = ''
        if samples_path == '':
            samples_path = input('Enter path to MCMC chain results: ')

        names = []
        if names == []:
            names = input('Enter parameter names: ')
            names = list(map(lambda x: x.strip(), names.split(',')))

        labels = []
        if labels == []:
            labels = input('Enter LaTeX labels for graph axes: ')
            labels = list(map(lambda x: x.strip(), labels.split(',')))

        # Set posterior plot output
        output = ''
        if output == '':
            output = input('Enter posterior plot output path: ')

        # Generate posterior plot
        posterior_plot(
            samples_path = samples_path,
            names = names,
            labels = labels,
            output = output
        )

    # Test action
    elif args.action == 'test':
        print('TESTING BRANCH')

        run1 = hod.nbar_loglike(
                cmass_M_min = 12.61718416755883,
                cmass_M_1 = 13.081194761572727,
                cmass_alpha = 1.143139485670338,
                cmass_M_0 = 12.097014571224458,
                cmass_sig_logm = 0.49826072097267643,
                wise_M_min = 13.354261137025665,
                wise_M_1 = 14.122981288643583,
                wise_alpha = 1.0169375422634406,
                wise_M_0 = 13.250846949648492,
                wise_sig_logm = 0.6880731244787067,
                R_ss = 0.0,
                R_cs = 0.0,
                R_sc = 0.0
        )

        print(f"Run 1: minimizer -439.354, current {run1}")

        run2 = hod.nbar_loglike(
                cmass_M_min = 12.61718416755883,
                cmass_M_1 = 13.081194761572727,
                cmass_alpha = 1.143139485670338,
                cmass_M_0 = 12.097014571224458,
                cmass_sig_logm = 0.49826072097267643,
                wise_M_min = 13.25975537,
                wise_M_1 = 14.00408087,
                wise_alpha = 1.03457798,
                wise_M_0 = 13.22919732,
                wise_sig_logm = 0.68298931,
                R_ss = 0.0,
                R_cs = 0.0,
                R_sc = 0.0
        )

        print(f"Run 2: minimizer -4106.86, current {run2}")

        # hod_params_list = [
        #     (13.04, 14.0, 0.950, 13.16, 0.43, 13.09, 13.775, 0.970, 13.44, 0.60),
        #     (13.04, 14.0, 0.975, 13.16, 0.47, 13.09, 13.775, 0.990, 13.44, 0.58),
        #     (12.94, 14.1, 0.950, 13.16, 0.43, 13.19, 13.775, 1.000, 13.64, 0.60),
        #     (13.04, 14.0, 0.975, 13.26, 0.48, 13.09, 13.675, 1.025, 13.54, 0.55),
        #     (13.14, 13.9, 1.000, 13.06, 0.53, 12.99, 13.875, 1.050, 13.44, 0.65),
        #     (12.94, 14.1, 1.025, 13.16, 0.43, 13.19, 13.775, 0.950, 13.64, 0.60),
        #     (13.04, 14.0, 1.050, 13.26, 0.48, 13.09, 13.675, 0.975, 13.54, 0.55),
        #     (13.14, 13.9, 0.950, 13.06, 0.53, 12.99, 13.875, 1.000, 13.44, 0.65),
        #     (12.94, 14.1, 0.975, 13.16, 0.43, 13.19, 13.775, 1.025, 13.64, 0.60),
        #     (13.04, 14.0, 1.000, 13.26, 0.48, 13.09, 13.675, 1.050, 13.54, 0.55)
        # ]

        # # idx = 6

        # t0 = time.time()
        # for idx in range(2):

        #     exclusion_ngmatched = hod.nbar_components(
        #         cmass_M_min = hod_params_list[idx][0],
        #         cmass_M_1 = hod_params_list[idx][1],
        #         cmass_alpha = hod_params_list[idx][2],
        #         cmass_M_0 = hod_params_list[idx][3],
        #         cmass_sig_logm = hod_params_list[idx][4],
        #         wise_M_min = hod_params_list[idx][5],
        #         wise_M_1 = hod_params_list[idx][6],
        #         wise_alpha = hod_params_list[idx][7],
        #         wise_M_0 = hod_params_list[idx][8],
        #         wise_sig_logm = hod_params_list[idx][9],
        #         R_ss = params["galaxy_corr"]["R_ss"]["val"],
        #         R_cs = params["galaxy_corr"]["R_cs"]["val"],
        #         R_sc = params["galaxy_corr"]["R_sc"]["val"]
        #     )

        #     print(f'Set {idx + 1}: NgMatched Exclusion ' + '-'*80)
        #     print(f'- Autocorrelation log-likelihood: {exclusion_ngmatched["auto_loglike"]}')
        #     print(f'- Cross-correlation log-likelihood: {exclusion_ngmatched["cross_loglike"]}')
        #     print(f'- Total log-likelihood: {exclusion_ngmatched["total_loglike"]}')
        #     print(f'- CMASS data nbar: {exclusion_ngmatched["cmass_nbar_data"]}')
        #     print(f'- CMASS model nbar: {exclusion_ngmatched["cmass_nbar_model"]}')
        #     print(f'- CMASS nbar correction: {exclusion_ngmatched["cmass_nbar_correction"]}')
        #     print(f'- WISE data nbar: {exclusion_ngmatched["wise_nbar_data"]}')
        #     print(f'- WISE model nbar: {exclusion_ngmatched["wise_nbar_model"]}')
        #     print(f'- WISE nbar correction: {exclusion_ngmatched["wise_nbar_correction"]}')
        #     print('time',time.time()-t0)
    # Test action
    elif args.action == 'test_vs_halofit':
        print('TESTING VS HALOFIT')
        
        hod = CMASS_WISE_HOD(
            cmass_redshift_file = cmass_redshift_file,
            wise_redshift_file = wise_redshift_file,
            data_file = data_file,
            covariance_file = covariance_file,
            params_file = params_file,
            cross_hod_model = VariableCorr,
            diag_covariance = False, 
            exclusion_model = 'NgMatched',
            exclusion_params = None,
            thetamin = 0.001,
            thetamax = 0.5,
            numtheta = 100
        )


        hod_params_list = [
            (13.04, 14.0, 0.950, 13.16, 0.43, 13.09, 13.775, 0.970, 13.44, 0.60)]
        # idx = 6

        idx = 0

        exclusion_ngmatched = hod.nbar_components(
           cmass_M_min = hod_params_list[idx][0],
           cmass_M_1 = hod_params_list[idx][1],
           cmass_alpha = hod_params_list[idx][2],
           cmass_M_0 = hod_params_list[idx][3],
           cmass_sig_logm = hod_params_list[idx][4],
           wise_M_min = hod_params_list[idx][5],
           wise_M_1 = hod_params_list[idx][6],
           wise_alpha = hod_params_list[idx][7],
           wise_M_0 = hod_params_list[idx][8],
           wise_sig_logm = hod_params_list[idx][9],
           R_ss = params["galaxy_corr"]["R_ss"]["val"],
           R_cs = params["galaxy_corr"]["R_cs"]["val"],
           R_sc = params["galaxy_corr"]["R_sc"]["val"]
        )
        

        b1 = hod.cross.halo_model_1.bias_effective_tracer
        b2 = hod.cross.halo_model_2.bias_effective_tracer

        dndz = hod.wise_redshift
        dndz_spec_all = hod.cmass_redshift
        zmin = hod.zmin
        zmax = hod.zmax
        dz = zmax-zmin

        theta = np.logspace(-3,np.log10(0.5),100)
        zarray = np.linspace(0.005,3.995,400)
        dndz_spec_all = np.array([zarray,dndz_spec_all]).T
        #dndz_spec_all[:,1][dndz_spec_all[:,0] < 0.45] = 0
        #dndz_spec_all[:,1][dndz_spec_all[:,0] > 0.5] = 0
        sys.path[0] = '/project/6033532/jptlawre'
        w_halofit = halofit_test.w_theta(b1, b2, dndz, dndz_spec_all, zmin, zmax, dz, theta, normed1=True)
        w_halomod = hod.cross.angular_corr_gal
        distance = 0.5 * (Planck15.comoving_distance(zmin)*0.6766 + Planck15.comoving_distance(zmax)*0.6766)
        R = theta * distance
        
        plt.figure()
        plt.plot(R*Planck15.H0.value/100.,w_halomod/w_halofit)
        plt.xlim(0,50)
        plt.ylim(0.9,1.1)
        plt.savefig('halofit_halomod_comparison_w_theta.png')
        
        r = hod.cross.halo_model_1.r
        xi_halomod = hod.cross.corr_cross-1.
        xi_halofit = halofit_test.xi(b1, b2, zmin, zmax, r)
        plt.figure()
        plt.plot(r, xi_halomod/xi_halofit)
        plt.xlim(0,50)
        plt.ylim(0.9,1.1)
        plt.savefig('halofit_halomod_comparison_xi.png')
        
        w_halofit = halofit_test.w_theta(b1, b1, dndz_spec_all, dndz_spec_all, zmin, zmax, dz, theta, normed1=False)
        w_halomod = hod.cmass_auto.angular_corr_gal

        plt.figure()
        plt.plot(R*Planck15.H0.value/100.,w_halomod/w_halofit)
        plt.xlim(0,50)
        plt.ylim(0.9,1.1)
        plt.savefig('halofit_halomod_comparison_w_theta_cmass_auto.png')

# ----------------------------------------------------------------------------------------------------------------------
