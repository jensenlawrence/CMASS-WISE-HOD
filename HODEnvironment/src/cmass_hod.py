# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# Basic imports
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

# Cosmology imports
from hmf import cached_quantity, parameter
from halomod.integrate_corr import AngularCF, angular_corr_gal
from halomod.cross_correlations import CrossCorrelations
from astropy.cosmology import Planck15
import copy
import time

# Custom imports
from get_model_info import get_model_params, get_model_dicts

# ----------------------------------------------------------------------------------------------------------------------
# AngularCrossCF Class
# ----------------------------------------------------------------------------------------------------------------------

class AngularCrossCF(CrossCorrelations):
    """
    Framework extension to angular correlation functions.
    """
    # Initialize class
    def __init__(self, p1=None, p2=None, theta_min=1e-3 * np.pi / 180.0, theta_max=np.pi / 180.0, theta_num=30,
                 theta_log=True, zmin=0.2, zmax=0.4, znum=100, logu_min=-4, logu_max=2.3, unum=100, check_p_norm=True,
                 p_of_z=True, exclusion_model=None, exclusion_params=None, **kwargs):
        """
        Initializes the AngularCrossCF class.

        Parameters
        ----------
        p1 : callable, optional
            The redshift distribution of the sample. This needs not be normalised to 1, as this will occur internally. May
            be either a function of radial distance [Mpc/h] or redshift. If a function of radial distance, `p_of_z` must be
            set to False. Default is a flat distribution in redshift.
        p2 : callable, optional
            See `p1`. This can optionally be a different function against which to cross-correlate. By default is
            equivalent to `p1`.
        theta_min, theta_max : float, optional
            min,max angular separations [Rad].
        theta_num : int, optional
            Number of steps in angular separation.
        theta_log : bool, optional
            Whether to use logspace for theta values.
        zmin, zmax : float, optional
            The redshift limits of the sample distribution. Note that this is in redshit, regardless of the value of
            `p_of_z`.
        znum : int, optional
            Number of steps in redshift grid.
        logu_min, logu_max : float, optional
            min, max of the log10 of radial separation grid [Mpc/h]. Must be large enough to let the integral over the 3D
            correlation function to converge.
        unum : int, optional
            Number of steps in the u grid.
        check_p_norm : bool, optional
            If False, cancels checking the normalisation of `p1` and `p2`.
        p_of_z : bool, optional
            Whether `p1` and `p2` are functions of redshift.
        kwargs : unpacked-dict
            Any keyword arguments passed down to :class:`halomod.HaloModel`.

        Returns
        -------
        None
        """
        super(AngularCrossCF, self).__init__(**kwargs)

        # if self.halo_model_1.z < zmin or self.halo_model_1.z > zmax:
        #     warnings.warn(
        #         f'Your specified redshift (z = {self.z}) is not within your selection function, z = ({zmin}, {zmax})'
        #     )

        if p1 is None:
            p1 = flat_z_dist(zmin, zmax)

        self.p1 = p1
        self.p2 = p2
        self.zmin = zmin
        self.zmax = zmax
        self.znum = znum
        self.logu_min = logu_min
        self.logu_max = logu_max
        self.unum = unum
        self.check_p_norm = check_p_norm
        self.p_of_z = p_of_z

        self.theta_min = theta_min
        self.theta_max = theta_max
        self.theta_num = theta_num
        self.theta_log = theta_log
        
        self.cosmo = self.halo_model_1.cosmo
        self.rnum = self.halo_model_1.rnum
        
        self.exclusion_model = exclusion_model
        self.exclusion_params = exclusion_params

    # p1 parameter
    @parameter("param")
    def p1(self, val):
        return val

    # p2 parameter
    @parameter("param")
    def p2(self, val):
        return val

    # p_of_z parameter
    @parameter("model")
    def p_of_z(self, val):
        return val

    # theta_min parameter
    @parameter("res")
    def theta_min(self, val):
        if val < 0:
            raise ValueError("theta_min must be > 0")
        return val

    # theta_max parameter
    @parameter("res")
    def theta_max(self, val):
        if val > 180.0:
            raise ValueError("theta_max must be < 180.0")
        return val

    # theta_num parameter
    @parameter("res")
    def theta_num(self, val):
        return val

    # theta_log parameter
    @parameter("res")
    def theta_log(self, val):
        return val

    # zmin parameter
    @parameter("param")
    def zmin(self, val):
        return val

    # zmax parameter
    @parameter("param")
    def zmax(self, val):
        return val

    # znum parameter
    @parameter("res")
    def znum(self, val):
        return val

    # logu_min parameter
    @parameter("res")
    def logu_min(self, val):
        return val

    # logu_max parameter
    @parameter("res")
    def logu_max(self, val):
        return val

    # unum parameter
    @parameter("res")
    def unum(self, val):
        return val

    # check_p_norm parameter
    @parameter("option")
    def check_p_norm(self, val):
        return val

    # Reshift distribution grid
    @cached_quantity
    def zvec(self):
        return np.linspace(self.zmin, self.zmax, self.znum)

    # Radial separation grid [Mpc/h]
    @cached_quantity
    def uvec(self):
        return np.logspace(self.logu_min, self.logu_max, self.unum)

    # Radial distance grid (corresponds to zvec) [Mpc/h]
    @cached_quantity
    def xvec(self):
        return self.cosmo.comoving_distance(self.zvec).value

    # Angular separations [Rad]
    @cached_quantity
    def theta(self):
        if self.theta_min > self.theta_max:
            raise ValueError("theta_min must be less than theta_max")

        if self.theta_log:
            return np.logspace(
                np.log10(self.theta_min), np.log10(self.theta_max), self.theta_num
            )
        else:
            return np.linspace(self.theta_min, self.theta_max, self.theta_num)

    # Physical separation grid [Mpc/h]
    @cached_quantity
    def r(self):
        rmin = np.sqrt(
            (10 ** self.logu_min) ** 2 + self.theta.min() ** 2 * self.xvec.min() ** 2
        )
        rmax = np.sqrt(
            (10 ** self.logu_max) ** 2 + self.theta.max() ** 2 * self.xvec.max() ** 2
        )
        return np.logspace(np.log10(rmin), np.log10(rmax), self.rnum)

    # Angular correlation function w(theta) from Blake+08, Eq. 33
    @cached_quantity
    def angular_corr_gal(self):
        def xi(r, z=self.halo_model_1.z):
            self.halo_model_1.z = z
            s = _spline(self.halo_model_1.r, self.corr_cross - 1.0, ext='zeros')
            return s(r)

        return angular_corr_gal(
            self.theta,
            xi,
            self.p1,
            self.zmin,
            self.zmax,
            self.logu_min,
            self.logu_max,
            znum=self.znum,
            unum=self.unum,
            p2=self.p2,
            check_p_norm=self.check_p_norm,
            cosmo=self.cosmo,
            p_of_z=self.p_of_z
        )

# ----------------------------------------------------------------------------------------------------------------------
# CMASS_HOD Class
# ----------------------------------------------------------------------------------------------------------------------

class CMASS_HOD:
    """
    HOD model for the autocorrelation of the galaxies observed at a redshift of z ~ 0.5 in the BOSS-CMASS galaxy survey.
    """
    # Initialize class
    def __init__(self, cmass_redshift_file, data_file, covariance_file, params_file,
                 cross_hod_model, diag_covariance=False, exclusion_model=None, exclusion_params=None):
        """
        Initializes the CMASS_WISE_HOD class.

        Parameters
        ----------
        cmass_reshift_file : str
            String representation of the path to the .txt file containing the BOSS-CMASS redshift distribution.
        data_file : str
            String representation of the path to the .txt file containing the CMASS autocorrelation and CMASS-WISE
            cross-correlation data as functions of R [Mpc/h].
        covariance_file : str
            String representation of the path to the .txt file containing the joint covariances matrix for the CMASS
            autocorrelation and CMASS-WISE cross-correlatoin data.
        params_file : str
            String representation of the path to the .json file containing the parameters for the BOSS-CMASS and WISE
            HOD models.
        cross_hod_model : AngularCrossCF
            HOD model for cross-correlations provided by an isntance of the AngularCrossCF class or any of its child
            classes.
        diag_covariance : bool, optional
            If True, only the diagonal values of the covariance matrix are used in calculations. If False, the full
            covariance matrix is used.

        thetamin : float, optional
            Gives an optional minimum angle to use, in radians. If no angle is given, theta is read from the data_file
        thetamax: float, optional
            Gives an optional maximum angle to use, in radians. If no angle is given, theta is read from the data_file
        numtheta: int, option
            Number of angles to use. If none is given, it is read from the data_file
        Returns
        -------
        None
        """
        # Initializing redshift distribution attributes
        self.cmass_redshift_file = cmass_redshift_file
        self.cmass_redshift = np.loadtxt(cmass_redshift_file, dtype=float)

        # Initializing data attribute
        self.data_file = data_file
        self.data = np.loadtxt(data_file, dtype=float)

        # Initializing covariance matrix attribute
        self.covariance_file = covariance_file 
        if diag_covariance:
            self.covariance = np.diag(np.diag(np.loadtxt(covariance_file, dtype=float)))
        else:
            self.covariance = np.loadtxt(covariance_file, dtype=float)

        # Initializing model attributes
        self.params_file = params_file 
        model_params = get_model_params(params_file)
        self.model_params = model_params

        cmass_model, _ = get_model_dicts(params_file)
        self.cmass_model = cmass_model

        # Initializing redshift limit attributes
        z = model_params['halo_params']['z']
        zmin = model_params['halo_params']['zmin']
        zmax = model_params['halo_params']['zmax']
        self.z = z
        self.zmin = zmin
        self.zmax = zmax

        # Initializing attributes for remaining parameters
        self.cross_hod_model = cross_hod_model 
        self.diag_covariance = diag_covariance 
        self.exclusion_model = exclusion_model
        self.exclusion_params = exclusion_params 

        # Calculating radial and angular values
        distance = Planck15.comoving_distance(0.5 * (zmin + zmax)).value * Planck15.H0.value/100.0
        thetas = self.data[:,0]/distance
        self.thetas = thetas

        # CMASS redshift calculations
        cmass_nz = np.loadtxt(cmass_redshift_file)
        cmass_z = np.linspace(0, 4.00, 401)
        cmass_zbin = 0.5 * (cmass_z[1:] + cmass_z[:-1])
        cmass_zfunc = _spline(cmass_zbin, cmass_nz)

        zrange = np.linspace(zmin, zmax, 100)
        chirange = Planck15.comoving_distance(zrange).value * Planck15.H0.value/100.0
        d_chirange = np.gradient(chirange)
        norm = np.sum(cmass_zfunc(zrange) * d_chirange * Planck15.H(zrange).value)
        cmass_zfunc_orig = _spline(Planck15.comoving_distance(cmass_zbin).value * Planck15.H0.value/100.0,
                                   cmass_nz * Planck15.H(cmass_zbin).value/norm)

        def cmass_zfunc(chi):
            out = np.zeros_like(chi) 
            out = cmass_zfunc_orig(chi) 
            out[chi < Planck15.comoving_distance(zmin).value * Planck15.H0.value/100.0] = 0 
            out[chi > Planck15.comoving_distance(zmax).value * Planck15.H0.value/100.0] = 0 
            return out

        norm = np.sum(cmass_zfunc(chirange) * d_chirange)
        cmass_zfunc = _spline(chirange, cmass_zfunc(chirange)/norm)
        #np.savetxt('chirange.txt',chirange)
        #np.savetxt('cmass_zfunc_chirange.txt',cmass_zfunc(chirange))
        #print('norm',norm)
        self.cmass_zfunc = cmass_zfunc
        
        # Summary of model attributes for comparison
        self.summary = (cmass_redshift_file, data_file, covariance_file, params_file,
                        cross_hod_model, diag_covariance, exclusion_model, exclusion_params)

    # Printable representation of class instance
    def __str__(self):
        """
        Provides a printable representation of an instance of the CMASS_WISE_HOD class.
        """
        rep_str = '-'*80
        rep_str += '\nInstance of the CMASS_HOD class.'
        rep_str += '\n' + '-'*80
        rep_str += '\nSources data from the files'
        rep_str += f'\n- Redshift: {self.cmass_redshift_file}'
        rep_str += f'\n- Data: {self.data_file}'
        rep_str += f'\n- Covariance: {self.covariance_file}'
        rep_str += f'\n- Model Parameters: {self.params_file}'
        rep_str += '\n' + '-'*80
        rep_str += f'\nUses the HOD model'
        rep_str += f'\n- Model 1: {self.cmass_model}'
        rep_str += '\n' + '-'*80

        return rep_str

    # Equivalence of class instances
    def __eq__(self, other):
        """
        Compares an instance of the CMASS_WISE_HOD class to any other object.

        Parameters
        ----------
        other : any
            Any other object being compared against.

        Returns
        -------
        are_equal : bool
            True if other is an instance of the CMASS_WISE_HOD class with identical parameters, and False otherwise.
        """
        are_equal = isinstance(other, CMASS_HOD) and (self.summary == other.summary)
        return are_equal

    # Autocorrelation function
    def cmass_auto(self):
        print('\nStarting cmass_auto\n')
        t0 = time.time()

        n_substeps = 100*self.data[:,0].size

        auto = AngularCF(
            p1 = self.cmass_zfunc,
            p_of_z = False,
            check_p_norm = True,
            theta_min = (np.min(self.data[:,0])-0.5 * np.gradient(self.data[:,0])[0])/(Planck15.comoving_distance(0.5 * (self.zmin + self.zmax)).value * Planck15.H0.value/100.0),
            theta_max = (np.max(self.data[:,0])+0.5 * np.gradient(self.data[:,0])[-1])/(Planck15.comoving_distance(0.5 * (self.zmin + self.zmax)).value * Planck15.H0.value/100.0),
            theta_num = n_substeps,
            theta_log = True,
            exclusion_model = self.exclusion_model,
            exclusion_params = self.exclusion_params,
            z = self.z,
            zmin = self.zmin,
            zmax = self.zmax,
            hod_params = self.cmass_model['hod_params'],
            hod_model = self.model_params['halo_params']['hod_model'],
            sigma_8 = self.model_params['halo_params']['sigma_8'],
            n = self.model_params['halo_params']['n'],
            hm_logk_min = self.model_params['halo_params']['hm_logk_min'],
            hm_logk_max = self.model_params['halo_params']['hm_logk_max'],
            hm_dlog10k = self.model_params['halo_params']['hm_dlog10k'],
            logu_min = -5,
            logu_max = 2.2,
            unum = 500
        )

        print(f'\ncmass_auto finished. t = {time.time() - t0} s\n')

        return auto

    # Calculate CMASS angular autocorrelation
    def corr_cmass_auto(self, update_cmass_params={}):
        """
        Executes Halomod's angular_corr_gal method on self.cmass_auto to compute the angular autocorrelation of the
        BOSS-CMASS HOD model.

        Parameters
        ----------
        update_cmass_params : dict, optional
            Dictionary containing parameters to udpate the BOSS-CMASS HOD model parameters.

        Returns
        -------
        cmass_auto_corr : array_like
            Array of calculated BOSS-CMASS autocorrelation values.
        """
        print('\nStarting corr_cmass_auto\n')
        t0 = time.time()

        # Get CMASS autocorrelation
        cmass_auto = self.cmass_auto()

        print(f'\nCalculated self.cmass_auto(). t = {time.time() - t0} s\n')

        # Update autocorrelation if updated CMASS parameters are provided
        if update_cmass_params != {}:
            cmass_auto.hod_params.update(update_cmass_params)

        print(f'\nUpdated HOD parameters. t = {time.time() - t0} s\n')

        # Calculate angular autocorrelation
        auto_corr = cmass_auto.angular_corr_gal_rsd

        print(f'\ncorr_cmass_auto finished. t = {time.time() - t0} s\n')
        return auto_corr

    # Components of the log-likelihood
    def loglike(self, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, wise_M_min, wise_M_1,
                wise_alpha, wise_M_0, wise_sig_logm, R_ss, R_cs, R_sc):
        """
        Calculates the individual components of the log-likelihood that the observed BOSS-CMASS data were
        produced by an HOD model with the BOSS-CMASS HOD model parameters.

        Parameters
        ----------
        cmass_M_min : float
            The minimum halo mass necessary for a CMASS dark matter halo to host a central galaxy.
        cmass_M_1 : float
            Mass parameter for CMASS satellite galaxies.
        cmass_alpha : float
            The exponent of the galaxy mass power law for CMASS galaxies.
        cmass_M_0 : float
            Mass parameter for CMASS satellite galaxies.
        cmass_sig_logm : float
            The step function smoothing parameter for WISE dark matter halos.
        R_ss : float
            The satellite-satellite correlation parameter for CMASS and WISE galaxies.
        R_cs : float
            The central-satellite correlation parameter for CMASS and WISE galaxies.
        R_sc : float
            The satellite-central correlation parameter for CMASS and WISE galaxies.

        Returns
        -------
        auto_loglike : dict
            The autocorrelation log-likelihood value.
        """
        print('\nStarting loglike\n')
        t0 = time.time()

        # Get data and covariance
        data = self.data[:,1]
        cov = self.covariance

        # Initialize parameter update dictionaries
        cmass_params = {
            'M_min': cmass_M_min,
            'M_1': cmass_M_1,
            'alpha': cmass_alpha,
            'M_0': cmass_M_0,
            'sig_logm': cmass_sig_logm
        }

        # Calculate CMASS autocorrelation and CMASS-WISE cross-correlation
        cmass_auto_corr_big = self.corr_cmass_auto(update_cmass_params=cmass_params)
        w_theory_binned = np.zeros(len(self.data[:,0]))
        bin_zero_min = np.log10(self.data[0,0])-0.5 * np.mean(np.gradient(np.log10(self.data[:,0])))
        bin_last_max = np.log10(self.data[-1,0])+0.5 * np.mean(np.gradient(np.log10(self.data[:,0])))
        bins_theta = np.logspace(bin_zero_min, bin_last_max, len(self.data)+1)
        cmass_auto_corr = np.zeros(len(bins_theta)-1)
        wspline = _spline(self.cmass_auto().theta, cmass_auto_corr_big)
        for i in range(len(w_theory_binned)):
            rp_big = np.linspace(bins_theta[i],bins_theta[i+1],1000)
            cmass_auto_corr[i] = (np.sum(2 * rp_big * (1 + wspline(rp_big))*np.gradient(rp_big))/((bins_theta[i+1]**2.-bins_theta[i]**2.))-1.)

        print(f'theta = {0.5 * (bins_theta[1:] + bins_theta[:-1])}') 
        print(f'auto = {cmass_auto_corr}')
        
        auto_chisq = np.linalg.multi_dot([data - cmass_auto_corr, np.linalg.inv(cov), data - cmass_auto_corr])
        auto_loglike = -0.5 * auto_chisq
        print(f'\nloglike finished. t = {time.time() - t0} s\n')

        # # Calculate CMASS autocorrelation and CMASS-WISE cross-correlation
        # cmass_auto_corr = self.corr_cmass_auto(update_cmass_params=cmass_params)
        # # Calculate autocorrelation-only log-likelihood
        # auto_chisq = np.linalg.multi_dot([data - cmass_auto_corr, np.linalg.inv(cov), data - cmass_auto_corr])
        # auto_loglike = -0.5 * auto_chisq

        return auto_loglike

    # Components of the nbar-weighted log-likelihood
    def nbar_loglike(self, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, wise_M_min, wise_M_1,
                     wise_alpha, wise_M_0, wise_sig_logm, R_ss, R_cs, R_sc):
        """
        Calculates the individual components of the nbar-weighted log-likelihood that the observed BOSS-CMASS data
        were produced by an HOD model with the BOSS-CMASS HOD model model parameters.

        Parameters
        ----------
        See `loglike_components`.

        Returns
        -------
        auto_loglike : dict
            The autocorrelation log-likelihood value weighted by the number density.
        """
        sig_nbar = 0.1

        # Get CMASS number densities and calculate CMASS number density correction
        cmass_nbar_data = self.model_params['nbar']['CMASS']
        cmass_nbar_model = self.cross.halo_model_1.mean_tracer_den
        cmass_nbar_correction = -0.5 * ((cmass_nbar_data - cmass_nbar_model)/(sig_nbar * cmass_nbar_data))**2

        # Get unweighted log-likelihood components
        loglike = self.loglike(cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, R_ss, R_cs, R_sc)

        return loglike + cmass_nbar_correction

# ----------------------------------------------------------------------------------------------------------------------