"""
Modules defining cross-correlated samples.

Has classes for both pure HOD cross-correlations
(i.e. number of cross-pairs) and for HaloModel-derived quantities
based on these cross-pairs.

To construct a :class:`CrossCorrelations` one need to specify the
halo models to be cross-correlated, and how they're correlated.

Examples
--------

Cross-correlating the same galaxy samples in different redshifts::

    >>> from halomod import HaloModel
    >>> from halomod.cross_correlations import CrossCorrelations, HODCross
    >>> cross = CrossCorrelations(cross_hod_model=ConstantCorr, halo_model_1_params=dict(z=1.0),
    >>>                           halo_model_2_params=dict(z=0.0))
    >>> pkcorr = cross.power_cross
"""

from halo_model import TracerHaloModel
from hmf import Component, Framework
from hmf._internals._framework import get_model_
from hmf._internals._cache import parameter, cached_quantity, subframework
from abc import ABC, abstractmethod
import numpy as np
from scipy import integrate as intg
import tools
from halo_exclusion_for_xcorr import DblEllipsoid, NgMatched, Exclusion, NoExclusion
from numpy import issubclass_


class HODCross(ABC, Component):
    """Provides methods necessary to compute cross-correlation pairs for HOD models."""

    _defaults = {}

    def __init__(self, hods, **model_params):
        super().__init__(**model_params)

        assert len(hods) == 2
        self.hods = hods

    @abstractmethod
    def R_ss(self, m):
        r"""The cross-correlation of numbers of pairs within a halo.

        Notes
        -----
        Defined by

        .. math:: \langle T_1 T_2 \rangle  = \langle T_1 \rangle \langle T_2 \rangle + \sigma_1 \sigma_2 R_{ss},

        where :math:`T` is the total amount of tracer in the halo's profile (i.e. not counting the
        central component, if this exists).
        """
        pass

    @abstractmethod
    def R_cs(self, m):
        r"""
        The cross-correlation of central-satellite pairs within a halo.

        Central from first hod, satellite from second.

        Notes
        -----
        Defined by

        .. math:: \langle T^c_1 T^s_2 \rangle  = \langle T^c_1 \rangle \langle T^s_2 \rangle + \sigma^c_1 \sigma^s_2 R_{cs},

        where :math:`T^s` is the total amount of tracer in the halo's profile (i.e. not counting the
        central component,if this exists).
        """
        pass

    @abstractmethod
    def R_sc(self, m):
        r"""
        The cross-correlation of satellite-central pairs within a halo.

        Central from second hod, Satellite from first.

        Notes
        -----
        Defined by

        .. math:: \langle T^s_1 T^c_2 \rangle  = \langle T^s_1 \rangle \langle T^c_2 \rangle + \sigma^s_1 \sigma^c_2 R_{sc},

        where :math:`T^s` is the total amount of tracer in the halo's profile (i.e. not counting
        the central component,if this exists).
        """
        pass

    @abstractmethod
    def self_pairs(self, m):
        r"""The expected number of cross-pairs at a separation of zero."""
        pass

    def ss_cross_pairs(self, m):
        r"""The average value of cross-pairs in a halo of mass m.

        Notes
        -----
        .. math:: `\langle T^s_1 T^s_2 \rangle - Q`"""
        h1, h2 = self.hods

        return (
            h1.satellite_occupation(m) * h2.satellite_occupation(m)
            + h1.sigma_satellite(m) * h2.sigma_satellite(m) * self.R_ss(m)
            - self.self_pairs(m)
        )

    def cs_cross_pairs(self, m):
        r"""The average value of cross-pairs in a halo of mass m.

        Notes
        -----
        .. math:: \langle T^c_1 T^s_2 \rangle.

        """
        h1, h2 = self.hods

        return h1.central_occupation(m) * h2.satellite_occupation(m) + h1.sigma_central(
            m
        ) * h2.sigma_satellite(m) * self.R_cs(m)

    def sc_cross_pairs(self, m):
        r"""The average value of cross-pairs in a halo of mass m,

        Notes
        -----
        .. math:: \langle T^s_1 T^c_2 \rangle
        """
        h1, h2 = self.hods

        return h2.central_occupation(m) * h1.satellite_occupation(m) + h2.sigma_central(
            m
        ) * h1.sigma_satellite(m) * self.R_sc(m)


class ConstantCorr(HODCross):
    """Correlation relation for constant cross-correlation pairs"""

    _defaults = {"R_ss": 0.0, "R_cs": 0.0, "R_sc": 0.0}

    def R_ss(self, m):
        return self.params["R_ss"]

    def R_cs(self, m):
        return self.params["R_cs"]

    def R_sc(self, m):
        return self.params["R_sc"]

    def self_pairs(self, m):
        """The expected number of cross-pairs at a separation of zero."""
        return 0


class CrossCorrelations(Framework):
    r"""
    The Framework for cross-correlations.

    This class generates two :class:`~halomod.halo_model.TracerHaloModel`,
    and calculates their cross-correlation according to the cross-correlation
    model given.

    Parameters
    ----------
    cross_hod_model : class
        Model for the HOD of cross correlation.
    cross_hod_params : dict
        Parameters for HOD used in cross-correlation.
    halo_model_1_params,halo_model_2_params : dict
        Parameters for the tracers used in cross-correlation.

    """

    def __init__(
        self,
        cross_hod_model,
        cross_hod_params={},
        halo_model_1_params={},
        halo_model_2_params={},
        exclusion_model=None,
        exclusion_params=None
    ):
        super().__init__()

        self.cross_hod_model = cross_hod_model
        self.cross_hod_params = cross_hod_params

        self._halo_model_1_params = halo_model_1_params
        self._halo_model_2_params = halo_model_2_params
        
        self.exclusion_model, self.exclusion_params = (
            exclusion_model,
            exclusion_params or {},
        )


    @parameter("model")
    def cross_hod_model(self, val):
        if not (isinstance(val, str) or np.issubclass_(val, HODCross)):
            raise ValueError(
                "cross_hod_model must be a subclass of cross_correlations.HODCross"
            )
        elif isinstance(val, str):
            return get_model_(val, "")
        else:
            return val

    @parameter("param")
    def cross_hod_params(self, val):
        return val
        
        
    @subframework
    def halo_model_1(self) -> TracerHaloModel:
        """Halo Model of the first tracer"""
        return TracerHaloModel(**self._halo_model_1_params)

    @subframework
    def halo_model_2(self) -> TracerHaloModel:
        """Halo Model of the second tracer"""
        return TracerHaloModel(**self._halo_model_2_params)
        
    @parameter("model")
    def exclusion_model(self, val):
        """A string identifier for the type of halo exclusion used (or None)."""
        if val is None:
            val = "NoExclusion"

        if issubclass_(val, Exclusion):
            return val
        else:
            return get_model_(val, "halomod.halo_exclusion_for_xcorr")
            
    @parameter("param")
    def exclusion_params(self, val):
        """Dictionary of parameters for the Exclusion model."""
        return val

    # ===========================================================================
    # Cross-correlations
    # ===========================================================================
    @cached_quantity
    def cross_hod(self):
        """HOD model of the cross-correlation"""
        return self.cross_hod_model(
            [self.halo_model_1.hod, self.halo_model_2.hod], **self.cross_hod_params
        )

    @cached_quantity
    def power_1h_cross_fnc(self):
        """Total 1-halo cross-power."""
        hm1, hm2 = self.halo_model_1, self.halo_model_2
        mask = np.logical_and(
            np.logical_and(
                np.logical_not(np.isnan(self.cross_hod.ss_cross_pairs(hm1.m))),
                np.logical_not(np.isnan(self.cross_hod.sc_cross_pairs(hm1.m))),
            ),
            np.logical_not(np.isnan(self.cross_hod.cs_cross_pairs(hm1.m))),
        )

        m = hm1.m[mask]
        u1 = hm1.tracer_profile_ukm[:, mask]
        u2 = hm2.tracer_profile_ukm[:, mask]

        integ = hm1.dndm[mask] * (
            u1 * u2 * self.cross_hod.ss_cross_pairs(m)
            + u1 * self.cross_hod.sc_cross_pairs(m)
            + u2 * self.cross_hod.cs_cross_pairs(m)
        )

        p = intg.simps(integ, m)

        p /= hm1.mean_tracer_den * hm2.mean_tracer_den
        return tools.ExtendedSpline(
            hm1.k, p, lower_func="power_law", upper_func="power_law"
        )

    @property
    def power_1h_cross(self):
        """Total 1-halo cross-power."""
        return self.power_1h_cross_fnc(self.halo_model_1.k_hm)

    @cached_quantity
    def corr_1h_cross_fnc(self):
        """The 1-halo term of the cross correlation"""
        corr = tools.hankel_transform(
            self.power_1h_cross_fnc, self.halo_model_1._r_table, "r"
        )
        return tools.ExtendedSpline(
            self.halo_model_1._r_table,
            corr,
            lower_func="power_law",
            upper_func=tools._zero,
        )

    @cached_quantity
    def corr_1h_cross(self):
        """The 1-halo term of the cross correlation"""
        return self.corr_1h_cross_fnc(self.halo_model_1.r)

    @cached_quantity
    def _power_halo_centres_fnc(self):
        """
        Power spectrum of halo centres, unbiased.
        Notes
        -----
        This defines the halo-centre power spectrum, which is a part of the 2-halo
        term calculation. Formally, we make the assumption that the halo-centre
        power spectrum is linearly biased, and this function returns
        .. math :: P^{hh}_c (k) /(b_1(m_1)b_2(m_2))
        """
        if self.halo_model_1.hc_spectrum == "filtered-lin":
            f = TopHat(None, None)
            p = self.halo_model_1.power * f.k_space(self.halo_model_1.k * 2.0)
            first_zero = np.where(p <= 0)[0][0]
            p[first_zero:] = 0
            return tools.ExtendedSpline(
                self.halo_model_1.k,
                p,
                lower_func=self.halo_model_1.linear_power_fnc,
                upper_func=tools._zero,
                match_lower=False,
            )
        elif self.halo_model_1.hc_spectrum == "filtered-nl":
            f = TopHat(None, None)
            p = self.halo_model_1.nonlinear_power * f.k_space(self.halo_model_1.k * 3.0)
            first_zero = np.where(p <= 0)[0][0]
            p[first_zero:] = 0
            return tools.ExtendedSpline(
                self.halo_model_1.k,
                p,
                lower_func=self.halo_model_1.nonlinear_power_fnc,
                upper_func=tools._zero,
                match_lower=False,
            )
        elif self.halo_model_1.hc_spectrum == "linear":
            return self.halo_model_1.linear_power_fnc
        elif self.halo_model_1.hc_spectrum == "nonlinear":
            return self.halo_model_1.nonlinear_power_fnc
        else:
            raise ValueError("hc_spectrum was specified incorrectly!")

    @cached_quantity
    def _power_2h_cross_primitive(self):
        """The 2-halo term of the cross-power spectrum."""
        import time
        t0 = time.time()
        hm1, hm2 = self.halo_model_1, self.halo_model_2

        u1 = hm1.tracer_profile_ukm[:, (hm1._tm & hm2._tm)]
        u2 = hm2.tracer_profile_ukm[:, (hm1._tm & hm2._tm)]
        
        if hm1.sd_bias_model is not None:
            bias1 = np.outer(hm1.sd_bias_correction, hm1.halo_bias)[:, (hm1._tm & hm2._tm)]
            bias2 = np.outer(hm2.sd_bias_correction, hm2.halo_bias)[:, (hm1._tm & hm2._tm)]
        else:
            bias1 = hm1.halo_bias[(hm1._tm & hm2._tm)]
            bias2 = hm2.halo_bias[(hm1._tm & hm2._tm)]
            
        inst = self.exclusion_model(
            m1=hm1.m[(hm1._tm & hm2._tm)],
            m2=hm2.m[(hm1._tm & hm2._tm)],
            density1 = hm1.total_occupation[(hm1._tm & hm2._tm)] * hm1.dndm[(hm1._tm & hm2._tm)],
            density2 = hm2.total_occupation[(hm1._tm & hm2._tm)] * hm2.dndm[(hm1._tm & hm2._tm)],
            Ifunc1=hm1.total_occupation[(hm1._tm & hm2._tm)]
            * hm1.dndm[(hm1._tm & hm2._tm)]
            * u1
            / hm1.mean_tracer_den,
            Ifunc2=hm2.total_occupation[(hm1._tm & hm2._tm)]
            * hm2.dndm[(hm1._tm & hm2._tm)]
            * u2
            / hm2.mean_tracer_den,
            bias1=bias1,
            bias2=bias2,
            r=hm1._r_table,
            delta_halo=hm1.halo_overdensity_mean,
            mean_density=hm1.mean_density0
        )
        
        if hasattr(inst, "density_mod"):
            self.__density_mod = inst.density_mod
        else:
            self.__density_mod = np.ones_like(hm1._r_table) * hm1.mean_tracer_den
            
        if hasattr(inst, "density_mod1"):
            self.__density_mod1 = inst.density_mod1
        else:
            self.__density_mod1 = np.ones_like(hm1._r_table) * hm1.mean_tracer_den

        if hasattr(inst, "density_mod2"):
            self.__density_mod2 = inst.density_mod2
        else:
            self.__density_mod2 = np.ones_like(hm2._r_table) * hm2.mean_tracer_den
        intg = inst.integrate()
        
        phh = self._power_halo_centres_fnc(hm1.k)

        if intg.ndim == 2:
            p = [
                tools.ExtendedSpline(
                    hm1.k,
                    x * phh,
                    lower_func=hm1.linear_power_fnc,
                    match_lower=True,
                    upper_func="power_law"
                    if (
                        self.halo_model_1.exclusion_model == NoExclusion
                        and "filtered" not in self.halo_model_1.hc_spectrum
                    )
                    else tools._zero,
                )
                for i, x in enumerate(intg)
            ]
        else:
            p = tools.ExtendedSpline(
                hm1.k,
                intg * phh,
                lower_func=hm1.linear_power_fnc,
                match_lower=True,
                upper_func="power_law"
                if (
                    self.halo_model_1.exclusion_model == NoExclusion
                    and "filtered" not in self.halo_model_1.hc_spectrum
                )
                else tools._zero,
            )
        #print('done',time.time()-t0)
        return p
        
    @property
    def power_2h_cross(self):
        """The 2-halo term of the tracer auto-power spectrum."""
        # If there's nothing modifying the scale-dependence, just return the original power.
        if self.halo_model_1.exclusion_model is NoExclusion and self.halo_model_1.sd_bias_model is None:
            return self._power_2h_cross_primitive(self.halo_model_1.k_hm)

        # Otherwise, first calculate the correlation function.
        out = tools.hankel_transform(
            self.corr_2h_cross_fnc, self.halo_model_1.k_hm, "k", h=0.001
        )

        # Everything below about k=1e-2 is essentially just the linear power biased,
        # and the hankel transform stops working at some small k.
        if np.any(self.halo_model_1.k_hm < 1e-2):
            warnings.warn(
                "power_2h_auto_tracer for k < 1e-2 is not computed directly, but "
                "is rather just the linear power * effective bias."
            )
            out[self.halo_model_1.k_hm < 1e-2] = (
                self.power[self.halo_model_1.k_hm < 1e-2] * self.halo_model_1.bias_effective_tracer * self.halo_model_2.bias_effective_tracer
            )

        return out
        
    @cached_quantity
    def corr_2h_cross_fnc(self):
        """A callable returning the 2-halo term of the tracer auto-correlation."""
        # Need to set h smaller here because this might need to be transformed back
        # to power.
        hm1, hm2 = self.halo_model_1, self.halo_model_2

        corr = tools.hankel_transform(
            self._power_2h_cross_primitive, hm1._r_table, "r", h=1e-4
        )

        # modify by the new density. This step is *extremely* sensitive to the exact
        # value of __density_mod at large
        # scales, where the ratio *should* be exactly 1.
        if hm1._r_table[-1] > 2 * hm1.halo_profile.halo_mass_to_radius(hm1.m[-1]):
            try:
                self.__density_mod1 *= hm1.mean_tracer_den / self.__density_mod1[-1]
            except TypeError:
                pass
        if hm2._r_table[-1] > 2 * hm2.halo_profile.halo_mass_to_radius(hm2.m[-1]):
            try:
                self.__density_mod2 *= hm2.mean_tracer_den / self.__density_mod2[-1]
            except TypeError:
                pass
        if hm1._r_table[-1] > 2 * hm1.halo_profile.halo_mass_to_radius(hm1.m[-1]):
            try:
                self.__density_mod *= np.sqrt(hm2.mean_tracer_den * hm1.mean_tracer_den) / self.__density_mod[-1]
            except TypeError:
                pass

        #corr = (self.__density_mod1 / hm1.mean_tracer_den) * (self.__density_mod2 / hm2.mean_tracer_den) * (1 + corr) - 1
        density_sq = ((self.__density_mod**2.)**2./np.sqrt(self.__density_mod1**2. * self.__density_mod2**2.))
        density_sq[(self.__density_mod == 0) & ((self.__density_mod1 == 0) | (self.__density_mod2 == 0))] = 0
        corr = density_sq / (hm1.mean_tracer_den * hm2.mean_tracer_den) * (1 + corr) - 1

        return tools.ExtendedSpline(
            hm1._r_table, corr, lower_func="power_law", upper_func=tools._zero
        )


    @cached_quantity
    def corr_2h_cross(self):
        """The 2-halo term of the cross-correlation."""
        return self.corr_2h_cross_fnc(self.halo_model_1.r)

    def power_cross_fnc(self, k):
        """Total tracer cross power spectrum."""
        return self.power_1h_cross_fnc(k) + self.power_2h_cross_fnc(k)

    @property
    def power_cross(self):
        """Total tracer cross power spectrum."""
        return self.power_cross_fnc(self.halo_model_1.k_hm)

    def corr_cross_fnc(self, r):
        """The tracer cross correlation function."""
        return self.corr_1h_cross_fnc(r) + self.corr_2h_cross_fnc(r) + 1

    @property
    def corr_cross(self):
        """The tracer cross correlation function."""
        print('IN CORR CROSS')
        return self.corr_cross_fnc(self.halo_model_1.r)