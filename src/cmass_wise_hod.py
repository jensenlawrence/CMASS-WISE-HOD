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

        if self.halo_model_1.z < zmin or self.halo_model_1.z > zmax:
            warnings.warn(
                f'Your specified redshift (z = {self.z}) is not within your selection function, z = ({zmin}, {zmax})'
            )

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
        def xi(r):
            s = _spline(self.halo_model_1.r, self.corr_cross - 1.0)
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
# CMASS_WISE_HOD Class
# ----------------------------------------------------------------------------------------------------------------------

class CMASS_WISE_HOD(AngularCrossCF):
    """
    HOD model for the cross-correlation of the galaxies observed at a redshift of z ~ 0.5 in the BOSS-CMASS and
    WISE galaxy surveys.
    """
    # Initialize class
    def __init__(self, cmass_redshift_file, wise_redshift_file, data_file, covariance_file, params_file,
                 cross_hod_model, diag_covariance=False, exclusion_model=None, exclusion_params=None):
        """
        Initializes the CMASS_WISE_HOD class.

        Parameters
        ----------
        cmass_reshift_file : str
            String representation of the path to the .txt file containing the BOSS-CMASS redshift distribution.
        wise_redshift_file : str
            String representation of the path to the .txt file containing the WISE redshift distribution.
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

        Returns
        -------
        None
        """
        # Initializing redshift distribution attributes
        self.cmass_redshift_file = cmass_redshift_file
        self.cmass_redshift = np.loadtxt(cmass_redshift_file, dtype=float)
        self.wise_redshift_file = wise_redshift_file
        self.wise_redshift = np.loadtxt(wise_redshift_file, dtype=float)

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

        cmass_model, wise_model = get_model_dicts(params_file)
        self.cmass_model = cmass_model 
        self.wise_model = wise_model 

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
        distance = Planck15.comoving_distance(z).value * Planck15.H0.value/100.0
        thetas = self.data[7:,0]/distance
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

        # WISE redshift calculations
        wise_zdist = np.loadtxt(wise_redshift_file)
        wise_zbin = wise_zdist[:,0]
        wise_nz = wise_zdist[:,1]
        wise_zfunc = _spline(wise_zbin, wise_nz)

        zrange = np.linspace(0, 4, 1000)
        chirange = Planck15.comoving_distance(zrange).value * Planck15.H0.value/100.0
        d_chirange = np.gradient(chirange)
        norm = np.sum(wise_zfunc(zrange) * d_chirange * Planck15.H(zrange).value)
        wise_zfunc = _spline(Planck15.comoving_distance(wise_zbin).value * Planck15.H0.value/100.0,
                             wise_nz * Planck15.H(wise_zbin).value/norm)

        # CMASS angular autocorrelation computation
        self.cmass_auto = AngularCF(
            p1 = cmass_zfunc,
            theta_min = np.min(thetas[3:]),
            theta_max = np.max(thetas[3:]),
            theta_num = len(thetas[3:]),
            theta_log = True,
            p_of_z = False,
            z = z,
            zmin = zmin,
            zmax = zmax,
            check_p_norm = False,
            hod_model = 'Zheng05',
            hod_params = cmass_model['hod_params'],
            logu_min = -5,
            logu_max = 2.2,
            unum = 500,
            exclusion_model = exclusion_model,
            exclusion_params = exclusion_params
        )

        # CMASS-WISE angular cross-correlation computation
        self.cross = AngularCrossCF(
            p1 = cmass_zfunc,
            p2 = wise_zfunc,
            theta_min = np.min(thetas),
            theta_max = np.max(thetas),
            theta_num = len(thetas),
            theta_log = True,
            p_of_z = False,
            zmin = zmin,
            zmax = zmax,
            cross_hod_model = cross_hod_model,
            check_p_norm = False,
            halo_model_1_params = cmass_model,
            halo_model_2_params = wise_model,
            logu_min = -5,
            logu_max = 2.2,
            unum = 500,
            exclusion_model = exclusion_model,
            exclusion_params = exclusion_params
        )

        # Summary of model attributes for comparison
        self.summary = (cmass_redshift_file, wise_redshift_file, data_file, covariance_file, params_file,
                        cross_hod_model, diag_covariance, exclusion_model, exclusion_params)

    # Printable representation of class instance
    def __str__(self):
        """
        Provides a printable representation of an instance of the CMASS_WISE_HOD class.
        """
        rep_str = '-'*80
        rep_str += '\nInstance of the CrossHOD class.'
        rep_str += '\n' + '-'*80
        rep_str += '\nSources data from the files'
        rep_str += f'\n- Redshift 1: {self.cmass_redshift_file}'
        rep_str += f'\n- Redshift 2: {self.wise_redshift_file}'
        rep_str += f'\n- Data: {self.data_file}'
        rep_str += f'\n- Covariance: {self.covariance_file}'
        rep_str += f'\n- Model Parameters: {self.params_file}'
        rep_str += '\n' + '-'*80
        rep_str += f'\nUses the HOD models'
        rep_str += f'\n- Model 1: {self.cmass_model}'
        rep_str += '\n'
        rep_str += f'\n- Model 2: {self.wise_model}'
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
        are_equal = isinstance(other, CMASS_WISE_HOD) and (self.summary == other.summary)
        return are_equal

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
        # Get CMASS autocorrelation
        cmass_auto = self.cmass_auto 

        # Update CMASS autocorrelation if updated parameters are provided
        if update_cmass_params != {}:
            print('-'*80)
            print('CMASS parameters before update', cmass_auto.hod_params)
            cmass_auto.hod_params.update(update_cmass_params)
            print('CMASS parameters after update', cmass_auto.hod_params)

        print('\n')
        print('cmass_auto attributes')
        print('- theta', cmass_auto.theta)
        print('- corr_auto_tracer_fnc', cmass_auto.corr_auto_tracer_fnc)
        print('- corr_1h_auto_tracer_fnc', cmass_auto.corr_1h_auto_tracer_fnc)
        print('- corr_2h_auto_tracer_fnc', cmass_auto.corr_2h_auto_tracer_fnc)
        print('- p1', cmass_auto.p1)
        print('- p2', cmass_auto.p2)
        print('- zmin', cmass_auto.zmin)
        print('- zmax', cmass_auto.zmax)
        print('- logu_min', cmass_auto.logu_min)
        print('- logu_max', cmass_auto.logu_max)
        print('- znum', cmass_auto.znum)
        print('- unum', cmass_auto.unum)
        print('- check_p_norm', cmass_auto.check_p_norm)
        print('- cosmo', cmass_auto.cosmo)
        print('- p_of_z', cmass_auto.p_of_z)

        # Calculate CMASS angular autocorrelation
        cmass_auto_corr = cmass_auto.angular_corr_gal

        return cmass_auto_corr

    # Calculate CMASS-WISE angular cross-correlation
    def corr_cross(self, update_cmass_params={}, update_wise_params={}):
        """
        Executes Halomod's angular_corr_gal method on self.cross to compute the angular cross-correlation of the
        BOSS-CMASS HOD model and the WISE HOD model.

        Parameters
        ----------
        update_cmass_params : dict, optional
            Dictionary containing parameters to udpate the BOSS-CMASS HOD model parameters.
        update_wise_params : dict, optional
            Dictionary containing parameters to udpate the WISE HOD model parameters.

        Returns
        -------
        cross_corr : array_like
            Array of calculated BOSS-CMASS and WISE cross-correlation values.
        """
        # Get cross-correlation
        cross = self.cross

        # Update cross-correlation if updated CMASS parameters are provided
        if update_cmass_params != {}:
            print('\n')
            print('CMASS parameters before update', cross.halo_model_1.hod_params)
            cross.halo_model_1.update(hod_params = update_cmass_params)
            print('CMASS parameters after update', cross.halo_model_1.hod_params)

        # Update cross-correlation if updated WISE parameters are provided
        if update_wise_params != {}:
            print('WISE parameters before update', cross.halo_model_2.hod_params)
            cross.halo_model_2.update(hod_params = update_wise_params)
            print('WISE parameters after update', cross.halo_model_2.hod_params)

        print('\n')
        print('cross attributes')
        print('- theta', cross.theta)
        print('- p1', cross.p1)
        print('- p2', cross.p2)
        print('- zmin', cross.zmin)
        print('- zmax', cross.zmax)
        print('- logu_min', cross.logu_min)
        print('- logu_max', cross.logu_max)
        print('- znum', cross.znum)
        print('- unum', cross.unum)
        print('- check_p_norm', cross.check_p_norm)
        print('- cosmo', cross.cosmo)
        print('- p_of_z', cross.p_of_z)

        # Calculate angular cross-correlation
        cross_corr = cross.angular_corr_gal
        return cross_corr

    # Components of the log-likelihood
    def loglike_components(self, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, wise_M_min, wise_M_1,
                           wise_alpha, wise_M_0, wise_sig_logm, R_ss, R_cs, R_sc):
        """
        Calculates the individual components of the log-likelihood that the observed BOSS-CMASS and WISE data were
        produced by an HOD model with the BOSS-CMASS HOD model and WISE HOD model parameters.

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
        wise_M_min : float
            The minimum halo mass necessary for a WISE dark matter halo to host a central galaxy.
        wise_M_1 : float
            Mass parameter for WISE satellite galaxies.
        wise_alpha : float
            The exponent of the galaxy mass power law for WISE galaxies.
        wise_M_0 : float
            Mass parameter for WISE satellite galaxies.
        wise_sig_logm : float
            The step function smoothing parameter for WISE dark matter halos.
        R_ss : float
            The satellite-satellite correlation parameter for CMASS and WISE galaxies.
        R_cs : float
            The central-satellite correlation parameter for CMASS and WISE galaxies.
        R_sc : float
            The satellite-central correlation parameter for CMASS and WISE galaxies.

        Returns
        -------
        loglike_dict : dict
            Dictionary containing the autocorrelation-only, cross-correlation-only, and total log-likelihood values.
        """
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
        wise_params = {
            'M_min': wise_M_min,
            'M_1': wise_M_1,
            'alpha': wise_alpha,
            'M_0': wise_M_0,
            'sig_logm': wise_sig_logm
        }

        # Calculate CMASS autocorrelation and CMASS-WISE cross-correlation
        cmass_auto_corr = self.corr_cmass_auto(update_cmass_params=cmass_params)
        cross_corr = self.corr_cross(update_cmass_params=cmass_params, update_wise_params=wise_params)

        # Calculate autocorrelation-only log-likelihood
        auto_chisq = np.linalg.multi_dot([data[:7] - cmass_auto_corr, np.linalg.inv(cov[:7,:7]), data[:7] - cmass_auto_corr])
        auto_loglike = -0.5 * auto_chisq

        # Calculate cross-correlation-only log-likelihood
        cross_chisq = np.linalg.multi_dot([data[7:] - cross_corr, np.linalg.inv(cov[7:,7:]), data[7:] - cross_corr])
        cross_loglike = -0.5 * cross_chisq

        # Calculate total log-likelihood
        total_corr = np.concatenate((cmass_auto_corr, cross_corr))
        total_chisq = np.linalg.multi_dot([data - total_corr, np.linalg.inv(cov), data - total_corr])
        total_loglike = -0.5 * total_chisq

        # Put results in dictionary
        loglike_dict = {
            'auto_loglike': auto_loglike,
            'cross_loglike': cross_loglike,
            'total_loglike': total_loglike
        }

        return loglike_dict

    # Total log-likelihood
    def loglike(self, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, wise_M_min, wise_M_1,
                wise_alpha, wise_M_0, wise_sig_logm, R_ss, R_cs, R_sc):
        """
        Calculates the total log-likelihood that the observed BOSS-CMASS and WISE data were produced by an
        HOD model with the BOSS-CMASS HOD model and WISE HOD model parameters.

        Parameters
        ----------
        See `loglike_components`.

        Returns
        -------
        total_loglike : float
            The total log-likelihood value.
        """
        loglike_dict = self.loglike_components(cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm,
                                               wise_M_min, wise_M_1, wise_alpha, wise_M_0, wise_sig_logm, R_ss,
                                               R_cs, R_sc)
        total_loglike = loglike_dict['total_loglike']
        return total_loglike

    # Components of the nbar-weighted log-likelihood
    def nbar_components(self, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, wise_M_min, wise_M_1,
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

        # Get CMASS number densities and calculate CMASS number density correction
        cmass_nbar_data = self.model_params['nbar']['CMASS']
        cmass_nbar_model = self.cross.halo_model_1.mean_tracer_den
        cmass_nbar_correction = -0.5 * ((cmass_nbar_data - cmass_nbar_model)/(sig_nbar * cmass_nbar_data))**2

        # Get WISE number densities and calculate WISE number density correction
        wise_nbar_data = self.model_params['nbar']['WISE']
        wise_nbar_model = self.cross.halo_model_2.mean_tracer_den 
        wise_nbar_correction = -0.5 * ((wise_nbar_data - wise_nbar_model)/(sig_nbar * wise_nbar_data))**2

        # Get unweighted log-likelihood components
        loglike_dict = self.loglike_components(cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm,
                                               wise_M_min, wise_M_1, wise_alpha, wise_M_0, wise_sig_logm, R_ss,
                                               R_cs, R_sc)

        # Update components dictionary
        loglike_dict['total_loglike'] += cmass_nbar_correction + wise_nbar_correction
        loglike_dict['cmass_nbar_data'] = cmass_nbar_data 
        loglike_dict['cmass_nbar_model'] = cmass_nbar_model
        loglike_dict['cmass_nbar_correction'] = cmass_nbar_correction
        loglike_dict['wise_nbar_data'] = wise_nbar_data 
        loglike_dict['wise_nbar_model'] = wise_nbar_model
        loglike_dict['wise_nbar_correction'] = wise_nbar_correction

        return loglike_dict

    # Total nbar-weighted log-likelihood
    def nbar_loglike(self, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, wise_M_min, wise_M_1,
                     wise_alpha, wise_M_0, wise_sig_logm, R_ss, R_cs, R_sc):
        """
        Calculates the total nbar-weighted log-likelihood that the observed BOSS-CMASS and WISE data were produced
        by an HOD model with the BOSS-CMASS HOD model and WISE HOD model parameters.

        Parameters
        ----------
        See `loglike_components`.

        Returns
        -------
        total_loglike : float
            The total nbar-weighted log-likelihood value.
        """
        loglike_dict = self.nbar_components(cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm,
                                            wise_M_min, wise_M_1, wise_alpha, wise_M_0, wise_sig_logm, R_ss,
                                            R_cs, R_sc)
        total_loglike = loglike_dict['total_loglike']
        return total_loglike

# ----------------------------------------------------------------------------------------------------------------------