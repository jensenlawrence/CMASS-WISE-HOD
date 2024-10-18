# ----------------------------------------------------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------------------------------------------------

# Basic imports
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline as _spline

# Cosmology imports
from hmf import cached_quantity, parameter
#from halomod.integrate_corr_beyond_limber import AngularCF, angular_corr_gal, angular_corr_gal_rsd
from integrate_corr_beyond_limber import AngularCF, angular_corr_gal, angular_corr_gal_rsd
from integrate_corr import angular_corr_gal as angular_corr_gal_orig
from cross_correlations import CrossCorrelations
from astropy.cosmology import Planck15
import copy
from hankel import HankelTransform
import time
import tools


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
    def __init__(self, p1=None, p2=None, thetas=None, theta_min=1e-3 * np.pi / 180.0, theta_max=np.pi / 180.0, theta_num=30,
                 theta_log=True, zmin=0.2, zmax=0.4, znum=18 , logu_min=-4, logu_max=2.3, unum=100, check_p_norm=True,
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
            optionally, give the array of thetas.
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

        if np.any(thetas):
            self.thetas = thetas
        else:
            self.theta_min = theta_min
            self.theta_max = theta_max
            self.theta_num = theta_num
            self.theta_log = theta_log
            self.thetas = None
        
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
        if np.any(self.thetas):
            return self.thetas
        else:
            if self.theta_min > self.theta_max:
                raise ValueError("theta_min must be less than theta_max")

            if self.theta_log:
                return np.logspace(
                    np.log10(self.theta_min), np.log10(self.theta_max), self.theta_num
                )
            else:
                return np.linspace(self.theta_min, self.theta_max, self.theta_num)
            
    @cached_quantity
    def corr_gg_0(self):
        return self.corr_cross - 1.0

    @property
    def corr_gg_2(self):
        t0 = time.time()
        ell = 2
        ht = HankelTransform(nu = ell + 0.5,
            N = 600,
            h = 0.005)
        k_hm = self.halo_model_1.k_hm
        #power_auto_tracer = self.power_auto_tracer
        #power_auto_tracer_fnc = tools.ExtendedSpline(k_hm, power_auto_tracer, lower_func='power_law',
        #    upper_func='power_law', k=1)
        print('got into corr_gg_2')
        print('self.power_1h_cross_fnc',self.power_1h_cross_fnc(k_hm))
        power_auto_tracer_interp = lambda k: self.power_1h_cross_fnc(k) * k**0.5 * np.sqrt(np.pi/2) * 1. /(2*np.pi**2.) * (1j)**ell
        xi_ell_1h = ht.transform(power_auto_tracer_interp, self.halo_model_1.r, ret_err=False)/self.halo_model_1.r**0.5
        #print('corr_gg_2: xi_ell_1h',time.time()-t0)
        power_auto_tracer = self.power_2h_cross
        power_2h_auto_tracer_fnc = tools.ExtendedSpline(k_hm, power_auto_tracer, lower_func='power_law',
            upper_func='power_law', k=1)
        power_auto_tracer_interp = lambda k: power_2h_auto_tracer_fnc(k) * k**0.5 * np.sqrt(np.pi/2) * 1. /(2*np.pi**2.) * (1j)**ell
        xi_ell_2h = ht.transform(power_auto_tracer_interp, self.halo_model_1.r, ret_err=False)/self.r**0.5
        #print('corr_gg_2: xi_ell_2h',time.time()-t0)
        cutoff_1h = 10.1
        xi_ell_1h_fixed = tools.ExtendedSpline(self.halo_model_1.r[self.halo_model_1.r < cutoff_1h], -1*np.real(xi_ell_1h)[self.halo_model_1.r < cutoff_1h], lower_func='power_law',
            upper_func='power_law',k=1)
        #print('time for hankel',time.time()-t0)
        xi_ell = np.real(xi_ell_2h) - xi_ell_1h_fixed(self.halo_model_1.r)
        #print('corr_gg_2: done',time.time()-t0)
        return xi_ell

    @property
    def corr_gg_4(self):
        t0 = time.time()
        ell = 4
        ht = HankelTransform(nu = ell + 0.5,
            N = 600,
            h = 0.005)
        k_hm = self.halo_model_1.k_hm
        #power_auto_tracer = self.power_auto_tracer
        #power_auto_tracer_fnc = tools.ExtendedSpline(k_hm, power_auto_tracer, lower_func='power_law',
        #    upper_func='power_law', k=1)
        power_auto_tracer_interp = lambda k: self.power_1h_cross_fnc(k) * k**0.5 * np.sqrt(np.pi/2) * 1. /(2*np.pi**2.) * (1j)**ell
        xi_ell_1h = ht.transform(power_auto_tracer_interp, self.r, ret_err=False)/self.halo_model_1.r**0.5
        #print('corr_gg_4: xi_ell_1h',time.time()-t0)
        power_auto_tracer = self.power_2h_cross
        power_2h_auto_tracer_fnc = tools.ExtendedSpline(k_hm, power_auto_tracer, lower_func='power_law',
            upper_func='power_law', k=1)
        power_auto_tracer_interp = lambda k: power_2h_auto_tracer_fnc(k) * k**0.5 * np.sqrt(np.pi/2) * 1. /(2*np.pi**2.) * (1j)**ell
        xi_ell_2h = ht.transform(power_auto_tracer_interp, self.halo_model_1.r, ret_err=False)/self.halo_model_1.r**0.5
        #print('corr_gg_4: xi_ell_2h',time.time()-t0)

        cutoff_1h = 15.
        xi_ell_1h_fixed = tools.ExtendedSpline(self.halo_model_1.r[self.halo_model_1.r < cutoff_1h], np.real(xi_ell_1h)[self.halo_model_1.r < cutoff_1h], lower_func='power_law',
            upper_func='power_law',k=1)
        #print('time for hankel',time.time()-t0)
        xi_ell = np.real(xi_ell_2h) + xi_ell_1h_fixed(self.halo_model_1.r)
        #print('corr_gg_4: done',time.time()-t0)

        return xi_ell


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
            #print('corr_cross',self.corr_cross)
            #print('z',self.halo_model_1.z)
            #self.halo_model_1.z=0.465
            #print('z',self.halo_model_1.z)
            #print('corr_cross again',self.corr_cross)
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
    @cached_quantity
    def angular_corr_matter(self):
        """
        The angular correlation function w(theta).

        From Blake+08, Eq. 33
        """

        def xi(r):
            #print('corr_mm',self.halo_model_1.corr_halofit_mm)
            #print('halofit',self.halo_model_1.nonlinear_power_fnc(0.1))
            s = _spline(self.halo_model_1.r, self.halo_model_1.corr_halofit_mm, ext='zeros')
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
            p_of_z=self.p_of_z,
        )

        
    @cached_quantity
    def angular_corr_gal_rsd(self):
        t0 = time.time()
        Om0 = self.cosmo.Om0
        Omz = (Om0* (1 + (self.halo_model_1.z))**3.)/((Om0) * (1 + (self.halo_model_1.z))**3. + (1 -Om0))
        f = Omz**0.55
        b1 = self.halo_model_1.bias_effective_tracer
        b2 = self.halo_model_2.bias_effective_tracer
        self.b1 = b1
        self.b2 = b2
        print('self.b1',self.b1)
        print('self.b2',self.b2)
        print('self.halo_model_1.z',self.halo_model_1.z)
        
        m1 = self.halo_model_1.mass_effective
        m2 = self.halo_model_2.mass_effective
        np.savetxt('dndm.txt',np.array([self.halo_model_1.m, self.halo_model_1.dndm, self.halo_model_1.central_occupation, self.halo_model_1.satellite_occupation]).T)
        #print('self.m1 in 364',self.m1)
        #print('self.m2 in 364',self.m2)
        self.m1 = m1
        self.m2 = m2
        #print('after linear biases',time.time()-t0)
        
        corrgg0 = self.corr_gg_0
        print('corrgg0',corrgg0)
        #print('corrg0',time.time()-t0)
        corrgg2 = self.corr_gg_2
        corrgg4 = self.corr_gg_4
        print('corrgg2',corrgg2)
        print('corrgg4',corrgg4)
        print('theta',self.theta)
        print('halo_model_1.r',self.halo_model_1.r)
        print('f/b1',f/b1)
        print('self.p1',self.p1)
        print('self.zmin',self.zmin)
        print('self.zmax',self.zmax)

        out = angular_corr_gal_rsd(
            self.theta,
            self.halo_model_1.r,
            corrgg0,
            corrgg2,
            corrgg4,
            f/b1,
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
            p_of_z=self.p_of_z,
            beta2 = f/b2
        )
        
        def xi(r, z=self.halo_model_1.z):
            #print('corr_cross',self.corr_cross)
            #print('z',self.halo_model_1.z)
            #self.halo_model_1.z=0.465
            #print('z',self.halo_model_1.z)
            #print('corr_cross again',self.corr_cross)
            self.halo_model_1.z = z
            s = _spline(self.halo_model_1.r, self.corr_cross - 1.0, ext='zeros')
            return s(r)
        
        #out2 = angular_corr_gal_orig(
        #    self.theta,
        #    xi,
        #    self.p1,
        #    self.zmin,
        #    self.zmax,
        #    self.logu_min,
        #    self.logu_max,
        #    znum=self.znum,
        #    unum=self.unum,
        #    p2=self.p2,
        #    check_p_norm=self.check_p_norm,
        #    cosmo=self.cosmo,
        #    p_of_z=self.p_of_z        )
        #print('after out',time.time()-t0)
        return out
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
                 magbias1, magbias2, magbias3, fiducial_cmass_bias, cross_hod_model, diag_covariance=False, exclusion_model=None, exclusion_params=None, min_bin=3, derived_name=None):
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
        magbias1: str
            String representation of the path to 1st magbias template (wise mu cross spec g). This one is multiplied by spec bias, so gets scaled in the code.
        magbias2: str
            String representation of the path to 2nd magbias template (wise g cross spec mu). This one is multiplied by wise bias, so needs all outputs from all redshift bins. Fixed in the code.
        magbias3: str
            String representation of the path to 3rd magbias template (mu cross mu)
        fiducial_cmass_bias: str
            fiducial_cmass_bias used to calculate the magnification terms
        cross_hod_model : AngularCrossCF
            HOD model for cross-correlations provided by an isntance of the AngularCrossCF class or any of its child
            classes.
        diag_covariance : bool, optional
            If True, only the diagonal values of the covariance matrix are used in calculations. If False, the full
            covariance matrix is used.
        min_bin : float, optional
            Minimum bin to use in autocorrelations (to account for fiber collisions). Minimum bin is 3 (correct
            for 0.45 < z < 0.50).
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
        self.wise_redshift_file = wise_redshift_file
        self.wise_redshift = np.loadtxt(wise_redshift_file, dtype=float)
        self.min_bin = min_bin

        # Initializing data attribute
        self.data_file = data_file
        self.data = np.loadtxt(data_file, dtype=float)
        
        # Intialize magbias templates
        self.mag_temp1 = np.loadtxt(magbias1)
        self.mag_temp2 = np.loadtxt(magbias2)
        self.mag_temp3 = np.loadtxt(magbias3)
        self.cmass_bias = fiducial_cmass_bias

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

        cmass_model, wise_model, corr_flag = get_model_dicts(params_file)
        self.cmass_model = cmass_model 
        self.wise_model = wise_model 
        self.corr_flag = corr_flag
        
        self.derived_name = derived_name

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
        thetas = self.data[:,0]/distance
        self.thetas = thetas

        # CMASS redshift calculations
        cmass_zbin_and_nz = np.loadtxt(cmass_redshift_file)
        cmass_zbin = cmass_zbin_and_nz[:,0]
        cmass_nz = cmass_zbin_and_nz[:,1]
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
        cmass_zfunc = _spline(chirange, cmass_zfunc(chirange)/norm, ext='zeros')

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
                             
        zrange = np.linspace(zmin, zmax, 100)
        chirange = Planck15.comoving_distance(zrange).value * Planck15.H0.value/100.0
        
        n_substeps = 3*self.data[self.min_bin:10,0].size
        
        theta_min = np.exp(np.min(np.log(self.data[self.min_bin:10,0]))-0.5 * np.gradient(np.log(self.data[self.min_bin:10,0]))[0])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0)
        theta_max = np.exp(np.max(np.log(self.data[self.min_bin:10,0]))+0.5 * np.gradient(np.log(self.data[self.min_bin:10,0]))[-1])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0)
        
        thetas_orig = np.logspace(np.log10(theta_min),np.log10(theta_max), n_substeps)
        if self.min_bin <= 3:
            thetas = np.concatenate((thetas_orig[:12],[thetas_orig[12],thetas_orig[15],thetas_orig[18]]))
        else:
            thetas = np.concatenate((thetas_orig[:12],[thetas_orig[12],thetas_orig[15]]))

        # CMASS angular autocorrelation computation
        self.cmass_auto = AngularCrossCF(
            p1 = cmass_zfunc,
            #theta_num = n_substeps,
            #theta_log = True,
            thetas = thetas,
            p_of_z = False,
            zmin = zmin,
            zmax = zmax,
            cross_hod_model = cross_hod_model,
            check_p_norm = True,
            halo_model_1_params = cmass_model,
            halo_model_2_params = cmass_model,
            logu_min = -5,
            logu_max = 2.2,
            unum = 500,
            exclusion_model = exclusion_model,
            exclusion_params = exclusion_params
        )
        
        n_substeps = 1*self.data[10:,0].size
        
        # CMASS-WISE angular cross-correlation computation
        self.cross = AngularCrossCF(
            p1 = cmass_zfunc,
            p2 = wise_zfunc,
            theta_min = np.exp(np.min(np.log(self.data[10:,0]))-0.5 * np.gradient(np.log(self.data[10:,0]))[0])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0),
            theta_max = np.exp(np.max(np.log(self.data[10:,0]))+0.5 * np.gradient(np.log(self.data[10:,0]))[-1])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0),
            #theta_min = np.min(self.data[7:,0])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0),
            #theta_max = np.max(self.data[7:,0])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0),
            theta_num = n_substeps,
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
        # Get cross-correlation
        t0 = time.time()
        cmass_auto = self.cmass_auto
        #print('self.cmass_auto',time.time()-t0)

        # Update cross-correlation if updated CMASS parameters are provided
        if update_cmass_params != {}:
            cmass_auto.halo_model_1.update(hod_params = update_cmass_params)
            cmass_auto.halo_model_2.update(hod_params = update_cmass_params)
        #print('update hod params',time.time()-t0)
        nbar = cmass_auto.halo_model_1.mean_tracer_den
        #print('mean tracer den',time.time()-t0)

        # Calculate angular cross-correlation
        auto_corr = cmass_auto.angular_corr_gal_rsd
        #print('done with corr_cmass_auto',time.time()-t0)
        #halofit = cmass_auto.angular_corr_gal
        return auto_corr, nbar
        
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
            cross.halo_model_1.update(hod_params = update_cmass_params)

        # Update cross-correlation if updated WISE parameters are provided
        if update_wise_params != {}:
            cross.halo_model_2.update(hod_params = update_wise_params)
            
        nbar = cross.halo_model_2.mean_tracer_den

        # Calculate angular cross-correlation
        cross_corr = cross.angular_corr_gal_rsd
        #halofit = cross.angular_corr_matter
        return cross_corr, nbar

    # Log-likelihood
    def loglike(self, cmass_M_min, cmass_M_1, cmass_alpha, cmass_M_0, cmass_sig_logm, wise_M_min, wise_M_1,
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
        t0 = time.time()
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
        
        if (self.corr_flag == "cmass_only") or (self.corr_flag == "combined") or (self.corr_flag == "wise_incompleteness"):

            cmass_auto_corr_big, cmass_nbar = self.corr_cmass_auto(update_cmass_params=cmass_params)
            #print('cmass auto corr',time.time()-t0)

            w_theory_binned = np.zeros(len(self.data[self.min_bin:10,0]))
            bin_zero_min = np.exp(np.min(np.log(self.data[self.min_bin:10,0]))-0.5 * np.gradient(np.log(self.data[self.min_bin:10,0]))[0])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0)
            bin_last_max = np.exp(np.max(np.log(self.data[self.min_bin:10,0]))+0.5 * np.gradient(np.log(self.data[self.min_bin:10,0]))[-1])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0)
            bins_theta = np.logspace(np.log10(bin_zero_min), np.log10(bin_last_max), len(self.data[self.min_bin:10])+1)
            cmass_auto_corr = np.zeros(len(bins_theta)-1)
            wspline = _spline(np.log(self.cmass_auto.theta), np.log(cmass_auto_corr_big))
            for i in range(len(w_theory_binned)):
                rp_big = np.linspace(bins_theta[i],bins_theta[i+1],1000)
                cmass_auto_corr[i] = (np.sum(2 * rp_big * (np.exp(wspline(np.log(rp_big))))*np.gradient(rp_big))/((bins_theta[i+1]**2.-bins_theta[i]**2.)))
                
            #print(f'theta = {0.5 * (bins_theta[1:] + bins_theta[:-1])}') 
            #print(f'auto = {cmass_auto_corr}')
            
            auto_chisq = np.linalg.multi_dot([data[self.min_bin:10] - cmass_auto_corr, np.linalg.inv(cov[self.min_bin:10,self.min_bin:10]), data[self.min_bin:10] - cmass_auto_corr])
            #print(f'auto = {data[:7]}')
            #print(f'cov = {cov[:7,:7]}')
            print(f'auto_chisq = {auto_chisq}')
            #print(f'auto fractional err = {np.sqrt(np.diag(cov[:7,:7]))/data[:7]}')
            auto_loglike = -0.5 * auto_chisq
            #print(f'\nloglike finished. t = {time.time() - t0} s\n')

        if (self.corr_flag == "cross_only") or (self.corr_flag == "combined") or (self.corr_flag == "wise_incompleteness"):

            cross_corr_big, wise_nbar = self.corr_cross(update_cmass_params=cmass_params, update_wise_params=wise_params)
            #print('cross corr',time.time()-t0)
            w_theory_binned = np.zeros(len(self.data[10:,0]))
            bin_zero_min = np.exp(np.min(np.log(self.data[10:,0]))-0.5 * np.gradient(np.log(self.data[10:,0]))[0])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0)
            bin_last_max = np.exp(np.max(np.log(self.data[10:,0]))+0.5 * np.gradient(np.log(self.data[10:,0]))[-1])/(Planck15.comoving_distance(self.z).value * Planck15.H0.value/100.0)
            bins_theta = np.logspace(np.log10(bin_zero_min), np.log10(bin_last_max), len(self.data[10:])+1)
            cross_corr = np.zeros(len(bins_theta)-1)
            halofit = np.zeros(len(bins_theta)-1)
            wspline = _spline(np.log(self.cross.theta), np.log(cross_corr_big))
            #halofit_spline = _spline(np.log(self.cross.theta),np.log(halofit_big))
            #halofit_spline2 = _spline(self.cross.theta,halofit_big)
            for i in range(len(w_theory_binned)):
                rp_big = np.linspace(bins_theta[i],bins_theta[i+1],1000)
                cross_corr[i] = (np.sum(2 * rp_big * (np.exp(wspline(np.log(rp_big))))*np.gradient(rp_big))/((bins_theta[i+1]**2.-bins_theta[i]**2.)))
                #halofit[i] = (np.sum(2 * rp_big * (np.exp(halofit_spline(np.log(rp_big))))*np.gradient(rp_big))/((bins_theta[i+1]**2.-bins_theta[i]**2.)))
                #halofit[i] = np.exp(halofit_spline(np.log(0.5 * (bins_theta[i] + bins_theta[i+1]))))
                #print(halofit[i])
                #halofit[i] = halofit_spline2(0.5 * (bins_theta[i] + bins_theta[i+1]))
                #print(halofit[i])
            #print(f'theta = {0.5 * (bins_theta[1:] + bins_theta[:-1])}') 
            #print(f'cross = {cross_corr}')
            real_cross = np.array([0.25835251, 0.18085699, 0.12428427, 0.08423749, 0.05826987, 0.04275561,
                 0.03118377, 0.02226584, 0.01566759, 0.01052752])
            #print('shitola',np.sqrt(np.diag(cov[7:,7:])))
            #print('CROSS DEV',(real_cross-cross_corr)/np.sqrt(np.diag(cov[7:,7:])))
            #print(f'halofit = {halofit}')
            
            #k = np.logspace(-2,1,100)
            #halofit_pk = self.cmass_auto.halo_model_1.nonlinear_power_fnc(k)
            #np.savetxt('/home/jptlawre/scratch/wca/halofit.txt',np.array([k,halofit_pk]).T)
            
            # Add magnification bias pieces
            cross_corr += self.mag_temp1 * self.cross.b1 / self.cmass_bias + self.mag_temp2 + self.mag_temp3
            print('mag bias pieces',self.mag_temp1 * self.cross.b1 / self.cmass_bias + self.mag_temp2 + self.mag_temp3)

            # Calculate cross-correlation-only log-likelihood
            # cross_chisq = np.linalg.multi_dot([data[7:] - cross_corr, np.linalg.inv(cov[7:,7:]), data[7:] - cross_corr])
            #cross_chisq = np.linalg.multi_dot([data[7:-1] - cross_corr[:-1], np.linalg.inv(cov[7:-1,7:-1]), data[7:-1] - cross_corr[:-1]])

            cross_chisq = np.linalg.multi_dot([data[10:] - cross_corr[:10], np.linalg.inv(cov[10:,10:]), data[10:] - cross_corr[:10]])
            cross_loglike = -0.5 * cross_chisq
            print(f'cross_chisq = {cross_chisq}')
            #f = open('cross_chisq_5x5.txt','a')
            #f.write('%10.5f\n' % cross_chisq)
            #f.close()

            #print(f'\ncross loglike finished. t = {time.time() - t0} s\n')


        if self.corr_flag == "cmass_only":
            total_loglike = auto_loglike

        if self.corr_flag == "cross_only":
            total_loglike =  cross_loglike

        if (self.corr_flag == "combined") or (self.corr_flag == "wise_incompleteness"):
            total_corr = np.concatenate((cmass_auto_corr, cross_corr[:10]))
            self.cmass_auto_corr = cmass_auto_corr
            self.cross_corr = cross_corr
            total_data = np.concatenate((data[self.min_bin:10], data[10:20]))
            print('auto data',data[self.min_bin:10])
            print('auto err',np.sqrt(np.diag(cov[self.min_bin:10,self.min_bin:10])))
            print('auto model',cmass_auto_corr)
            real_auto = np.array([2.68455091, 1.60922933, 1.02980922, 0.72910976, 0.51656879, 0.35955015,
                    0.2388804 ])
            #print('AUTO DEV',(real_auto-cmass_auto_corr)/np.sqrt(np.diag(cov[:7])))

            print('cross data',data[10:])
            print('cross err',np.sqrt(np.diag(cov[10:,10:])))
            print('cross model',cross_corr[:10])
            print('cmass mean mass',self.cross.m1)
            print('wise mean mass',self.cross.m2)
            #print('auto cov',cov[:7,:7])
            #print('cross cov',cov[9:17,9:17])
            #print(f'theta = {0.5 * (bins_theta[1:] + bins_theta[:-1])}') 
            total_cov = np.zeros((20-self.min_bin,20-self.min_bin))
            total_cov[:10-self.min_bin,:10-self.min_bin] = cov[self.min_bin:10,self.min_bin:10]
            total_cov[10-self.min_bin:,10-self.min_bin:] = cov[10:20,10:20]
            np.savetxt('total_cov.txt',total_cov)
            total_chisq = np.linalg.multi_dot([total_data - total_corr, np.linalg.inv(total_cov), total_data - total_corr])
            print('total_chisq',total_chisq)
            total_loglike = -0.5 * total_chisq
        #print('total time',time.time()-t0)

        return cmass_nbar, wise_nbar, total_loglike


# ----------------------------------------------------------------------------------------------------------------------
