# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# Basic imports
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

# Cosmology imports
from hmf import cached_quantity, parameter
from halomod import TracerHaloModel
from halomod.integrate_corr import AngularCF, angular_corr_gal
from halomod.cross_correlations import CrossCorrelations, HODCross
from astropy.cosmology import Planck15

# Custom imports
from get_model_info import get_model_params, get_model_dicts

# ----------------------------------------------------------------------------------------------------------------------
# AngularCrossCF Class
# ----------------------------------------------------------------------------------------------------------------------

# Class for computing angular cross-correlation
class AngularCrossCF(CrossCorrelations):
    """
    Framework extension to angular correlation functions.
    """
    def __init__(self, p1=None, p2=None, theta_min=1e-3 * np.pi / 180.0, theta_max=np.pi / 180.0, theta_num=30,
                 theta_log=True, zmin=0.2, zmax=0.4, znum=100, logu_min=-4, logu_max=2.3, unum=100, check_p_norm=True,
                 p_of_z=True, exclusion_model=None, exclusion_params=None, **kwargs):
        """
        __init__ : NoneType, NoneType, float, float, int, bool, float, float, int, float, float, int, bool,
                   bool, NoneType, NoneType, any -> AngularCrossCF
            Defines the attributes of the AngularCrossCF class.

        p1 : callable, optional
            The redshift distribution of the sample. This needs not be normalised to 1, as this will occur internally. May
            be either a function of radial distance [Mpc/h] or redshift. If a function of radial distance, `p_of_z` must be
            set to False. Default is a flat distribution in redshift.

        p2 : callable, optional
            See `p1`. This can optionally be a different function against which to cross-correlate. By default is
            equivalent to `p1`.

        theta_min, theta_max : float, optional
            min,max angular separations [Rad]

        theta_num : int, optional
            Number of steps in angular separation

        theta_log : bool, optional
            Whether to use logspace for theta values

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
        """
        super(AngularCrossCF, self).__init__(**kwargs)
        #print('self z',self.z)
        print('self halo model 1 z',self.halo_model_1.z)
        print('self halo model 2 z',self.halo_model_2.z)

        if self.halo_model_1.z < zmin or self.halo_model_1.z > zmax:
            warnings.warn(
                "Your specified redshift (z=%s) is not within your selection function, z=(%s,%s)"
                % (self.z, zmin, zmax)
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

    @parameter("param")
    def p1(self, val):
        return val

    @parameter("param")
    def p2(self, val):
        return val

    @parameter("model")
    def p_of_z(self, val):
        return val

    @parameter("res")
    def theta_min(self, val):
        if val < 0:
            raise ValueError("theta_min must be > 0")
        return val

    @parameter("res")
    def theta_max(self, val):
        if val > 180.0:
            raise ValueError("theta_max must be < 180.0")
        return val

    @parameter("res")
    def theta_num(self, val):
        return val

    @parameter("res")
    def theta_log(self, val):
        return val

    @parameter("param")
    def zmin(self, val):
        return val

    @parameter("param")
    def zmax(self, val):
        return val

    @parameter("res")
    def znum(self, val):
        return val

    @parameter("res")
    def logu_min(self, val):
        return val

    @parameter("res")
    def logu_max(self, val):
        return val

    @parameter("res")
    def unum(self, val):
        return val

    @parameter("option")
    def check_p_norm(self, val):
        return val

    @cached_quantity
    def zvec(self):
        """
        Redshift distribution grid.
        """
        return np.linspace(self.zmin, self.zmax, self.znum)

    @cached_quantity
    def uvec(self):
        """
        Radial separation grid [Mpc/h].
        """
        return np.logspace(self.logu_min, self.logu_max, self.unum)

    @cached_quantity
    def xvec(self):
        """
        Radial distance grid (corresponds to zvec) [Mpc/h].
        """
        return self.cosmo.comoving_distance(self.zvec).value

    @cached_quantity
    def theta(self):
        """
        Angular separations, [Rad].
        """
        if self.theta_min > self.theta_max:
            raise ValueError("theta_min must be less than theta_max")

        if self.theta_log:
            return np.logspace(
                np.log10(self.theta_min), np.log10(self.theta_max), self.theta_num
            )
        else:
            return np.linspace(self.theta_min, self.theta_max, self.theta_num)

    @cached_quantity
    def r(self):
        """
        Physical separation grid [Mpc/h].
        """
        rmin = np.sqrt(
            (10 ** self.logu_min) ** 2 + self.theta.min() ** 2 * self.xvec.min() ** 2
        )
        rmax = np.sqrt(
            (10 ** self.logu_max) ** 2 + self.theta.max() ** 2 * self.xvec.max() ** 2
        )
        return np.logspace(np.log10(rmin), np.log10(rmax), self.rnum)

    @cached_quantity
    def angular_corr_gal(self):
        """
        The angular correlation function w(theta) from Blake+08, Eq. 33.
        """

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
# CrossHOD Class
# ----------------------------------------------------------------------------------------------------------------------

# Class for cross-correlated HODs
class CrossHOD(AngularCrossCF):
    """
    Class for constructing an HOD model for the cross-correlation of the galaxies observed at a redshift of
    z ~ 0.5 in the BOSS-CMASS and WISE galaxy surveys.
    """
    # Initialize class
    def __init__(self, cmass_redshift_file, wise_redshift_file, data_file, covariance_file, params_file, cross_hod_model,
                 diag_cov=False, exclusion_model=None, exclusion_params=None):
        """
        __init__ : self, str, str, str, str, str, AngularCrossCF, bool, NoneType, NoneType -> CrossHOD
            Defines the attributes of the CrossHOD class.

        cmass_redshift_file : str
            String representation of the path to the .txt BOSS-CMASS redshift distribution file.

        wise_redshift_file : str
            String representation of the path to the .txt WISE redshift distribution file.

        data_file : str
            String representation of the path to the .txt file containing the CMASS autocorrelation and CMASS-WISE
            cross-correlation data as functions of R.

        covariance_file : str
            String representation of the path to the .txt file containing the covariance matrix describing the
            CMASS autocorrelation and CMASS-WISE cross-correlation.

        params_file : str
            String representation of the path to the .json file containing the parameters for the BOSS-CMASS and
            WISE HOD models.

        cross_hod_model : AngularCrossCF
            Cross-correlated HOD model provided by an instance of the AngularCrossCF class.

        diag_cov : bool
            Optional argument that determines whether the full covariance matrix is used, or only the diagonal
            entries of the covariance matrix are used.
            Default value is False.
        """
        # Initializing redshift distributions
        self.cmass_redshift_file = cmass_redshift_file
        self.wise_redshift_file = wise_redshift_file

        # Initializing data
        self.data_file = data_file
        self.data = np.loadtxt(data_file)

        # Initializing covariance
        self.covariance_file = covariance_file
        if diag_cov:
            self.covariance = np.diag(np.diag(np.loadtxt(covariance_file)))
        else:
            self.covariance = np.loadtxt(covariance_file)

        # Initializing models
        self.params_file = params_file
        model_params = get_model_params(params_file)
        self.model_params = model_params

        z = model_params['halo_params']['z']
        zmin = model_params['halo_params']['zmin']
        zmax = model_params['halo_params']['zmax']
        self.z = z
        self.zmin = zmin
        self.zmax = zmax

        cmass_model, wise_model = get_model_dicts(params_file)
        self.cmass_model = cmass_model
        self.wise_model = wise_model

        # Other model parameters
        self.cross_hod_model = cross_hod_model
        self.diag_cov = diag_cov
        self.exclusion_model = exclusion_model
        self.exclusion_params = exclusion_params

        # Calculating distances and theta values
        distance = Planck15.comoving_distance(z).value * Planck15.H0.value/100.0
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
            zmin = zmin,
            zmax = zmax,
            z = self.cmass_model['z'],
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

        # Summary for comparison to other models
        self.summary = (cmass_redshift_file, wise_redshift_file, data_file, covariance_file, params_file, cross_hod_model,
                        diag_cov, exclusion_model, exclusion_params)

    # Print representation
    def __str__(self):
        """
        __str__ : self -> str
            Provides a string representation of a given instance of the CrossHOD class.
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

    # Class equivalence
    def __eq__(self, other):
        """
        __eq__ : self, any -> bool
            Allows for the comparison of an instance of the CrossHOD class to another object.
            Returns True if both are instances of the CrossHOD class with identical properties, and
            False otherwise.

        other : any
            Any object against which a given instance of the CrossHOD class is compared.
        """
        return isinstance(other, CrossHOD) and (self.summary == other.summary)

    # CMASS angular autocorrelation
    def corr_cmass_auto(self):
        """
        corr_cmass_auto : self -> array-like
            Executes Halomod's angular_corr_gal method to compute the angular autocorrelation of the CMASS HOD model.
        """
        #corr auto tracer
        cmass_auto = self.cmass_auto
        return cmass_auto.angular_corr_gal
    
    # CMASS-WISE angular cross-correlation
    def corr_cross(self):
        """
        corr_cross : self -> array-like
            Executes Halomod's angular_corr_gal method to compute the angular cross-correlation of the CMASS and
            WISE HOD models.
        """
        cross = self.cross
        return cross.angular_corr_gal

    # Total model log-likelihood
    def likelihood(self, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm,
                   wise_M_min, wise_M_1, wise_alpha, wise_M_0, wise_sig_logm, R_ss, R_cs, R_sc):
        """
        likelihood : self, float, float, float, float, float, float, float, float, float, float, float
                     float, float -> float
            Computes the log-likelihood that an HOD model for the CMASS and WISE galaxies with the given parameters
            matches the data.

        cmass_M_min : float
            The minimum halo mass necessary for a CMASS dark matter halo to host a central galaxy.

        cmass_M_1 : float
            A mass parameter for CMASS satellite galaxies.

        cmass_alpha : float
            The exponent of the galaxy mass power law for CMASS galaxies.

        cmass_M_0 : float
            A mass paramter for CMASS satellite galaxies.

        cmass_sig_logm : float
            The step function smoothing parameter for CMASS dark matter halos.

        wise_M_min : float
            The minimum halo mass necessary for a WISE dark matter halo to host a central galaxy.

        wise_M_1 : float
            A mass parameter for WISE satellite galaxies.

        wise_alpha : float
            The exponent of the galaxy mass power law for WISE galaxies.

        wise_M_0 : float
            A mass paramter for WISE satellite galaxies.

        wise_sig_logm : float
            The step function smoothing parameter for WISE dark matter halos.

        R_ss : float
            The satellite-satellite correlation parameter for CMASS and WISE galaxies.

        R_cs : float
            The central-satellite correlation parameter for CMASS and WISE galaxies.

        R_sc : float
            The satellite-central correlation parameter for CMASS and WISE galaxies.
        """
        data = self.data[:,1]
        cov = self.covariance

        cmass_auto = self.cmass_auto
        cmass_auto.hod_params.update(
            {
                'M_min': cmass_M_min,
                'M_1': cmass_M_1,
                'alpha': cmass_alpha,
                'M_0': cmass_M_0,
                'sig_logm': cmass_sig_logm
            }
        )
        cmass_auto_corr = cmass_auto.angular_corr_gal

        cross = self.cross
        cross.halo_model_1.update(
            hod_params = {
                'M_min': cmass_M_min,
                'M_1': cmass_M_1,
                'alpha': cmass_alpha,
                'M_0': cmass_M_0,
                'sig_logm': cmass_sig_logm
            }
        )
        cross.halo_model_2.update(
            hod_params = {
                'M_min': wise_M_min,
                'M_1': wise_M_1,
                'alpha': wise_alpha,
                'M_0': wise_M_0,
                'sig_logm': wise_sig_logm
            }
        )
        cross_corr = cross.angular_corr_gal

        total_corr = np.concatenate((cmass_auto_corr, cross_corr))
        total_chisq = np.linalg.multi_dot([data - total_corr, np.linalg.inv(cov), data - total_corr])
        total_loglike = -0.5 * total_chisq

        return total_loglike

    # Total model log-likelihood with number density weighting
    def nbar_likelihood(self, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm,
                        wise_M_min, wise_M_1, wise_alpha, wise_M_0, wise_sig_logm, R_ss, R_cs, R_sc):
        """
        nbar_likelihood : self, float, float, float, float, float, float, float, float, float, float, float
                          float, float -> float
            Computes the log-likelihood that an HOD model for the CMASS and WISE galaxies with the given parameters
            matches the data, using the CMASS and WISE galaxy number densities as an additional constraint.

        Parameters are the same as likelihood.
        """
        sig_nbar = 0.1

        cmass_nbar_data = self.model_params['nbar']['CMASS']
        cmass_nbar_model = self.cross.halo_model_1.mean_tracer_den
        cmass_nbar_correction = -0.5 * ((cmass_nbar_data - cmass_nbar_model)/(sig_nbar * cmass_nbar_data))**2

        wise_nbar_data = self.model_params['nbar']['WISE']
        wise_nbar_model = self.cross.halo_model_2.mean_tracer_den 
        wise_nbar_correction = -0.5 * ((wise_nbar_data - wise_nbar_model)/(sig_nbar * wise_nbar_data))**2

        loglike = self.likelihood(cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm,
                                  wise_M_min, wise_M_1, wise_alpha, wise_M_0, wise_sig_logm, R_ss, R_cs, R_sc)

        return loglike + cmass_nbar_correction + wise_nbar_correction
 
# ----------------------------------------------------------------------------------------------------------------------