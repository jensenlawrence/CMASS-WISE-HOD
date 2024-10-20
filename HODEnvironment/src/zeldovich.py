#
from __future__ import print_function,division

import numpy
#import kernels
from mcfit import SphericalBessel as sph
#mcfit multiplies by sqrt(2/pi)*x**2 to the function. 
#Divide the funciton by this to get the correct form 

from scipy.integrate import quad
from scipy.interpolate import InterpolatedUnivariateSpline as interpolate
from scipy.misc import derivative
import sys

class Zeldovich:
    '''
    Class to evaluate the Zeldovich power spectrum, given a linear power
    spectrum k, p [in compatible units, e.g. h/Mpc and (Mpc/h)^3].
    This can be used as the basis for both the HaloZeldovich and ZEFT
    approximations, and for auto and cross-spectra.
    See:
    Modi, White & Vlah, arXiv:1706.03173
    https://arxiv.org/abs/1706.03173
    for more information.
    To main method is make_table(), which creates the table of power spectra
    components.  The order is k, ZA, b1, b2, b1sq, b2sq, b1b1
    Convenience functions are provided for common calls.
    '''
    def __init__(self, k, p):
        '''k,p are the linear theory power spectra in compatible units,
        e.g. h/Mpc and (Mpc/h)^3.'''
        self.kp    = k
        self.p     = p
        self.ilpk  = self.loginterp(k, p)
        self.renorm=numpy.sqrt(numpy.pi/2.) #mcfit normaliztion
        self.tpi2  = 2*numpy.pi**2.
        self.qt    = numpy.logspace(-5, 5, int(1e4))
        self.kint  = numpy.logspace(-5, 5, int(1e4))
        self.jn    = 10 #number of bessels to sum over
        self.pktable=None
        self.setup()
        #
    def setup(self):
        '''
        Create X_L, Y_L, xi_L, U1_L \& 0lag sigma.
        '''
        self.xi0lag = self.xi0lin0() 
        self.qv, xi0v = self.xi0lin()
        xi2v = self.xi2lin()[1]
        self.corlin = self.corr()[1]
        self.Ulin = self.u10lin()[1]
        #
        self.Xlin = 2/3.*(self.xi0lag - xi0v - xi2v)
        ylinv = 2*xi2v
        #Since we divide by ylin, check for zeros
        mask = (ylinv == 0)
        ylinv[mask] = interpolate(self.qv[~mask], ylinv[~mask])(self.qv[mask])
        self.Ylin = ylinv
        self.XYlin = (self.Xlin + self.Ylin)
        self.sigma = self.XYlin[-1]
        self.yq = (1*self.Ylin/self.qv)
    ### Interpolate functions in log-sapce beyond the limits
    def loginterp(self, x, y, yint = None, side = "both",\
                  lorder = 15, rorder = 15, lp = 1, rp = -1, \
                  ldx = 1e-6, rdx = 1e-6):
        '''
        Extrapolate function by evaluating a log-index of left & right side
        '''
        if yint is None:
            yint = interpolate(x, y, k = 5)
        if side == "both":
            side = "lr"
            l =lp
            r =rp
        lneff = derivative(yint, x[l], dx = x[l]*ldx, order = lorder)*x[l]/y[l]
        rneff = derivative(yint, x[r], dx = x[r]*rdx, order = rorder)*x[r]/y[r]
        print('Log index on left & right edges are = ', lneff, rneff)
        #
        xl = numpy.logspace(-18, numpy.log10(x[l]), 10**6)
        xr = numpy.logspace(numpy.log10(x[r]), 10., 10**6)
        yl = y[l]*(xl/x[l])**lneff
        yr = y[r]*(xr/x[r])**rneff
        #
        xint = x[l+1:r].copy()
        yint = y[l+1:r].copy()
        if side.find("l") > -1:
            xint = numpy.concatenate((xl, xint))
            yint = numpy.concatenate((yl, yint))
        if side.find("r") > -1:
            xint = numpy.concatenate((xint, xr))
            yint = numpy.concatenate((yint, yr))
        yint2 = interpolate(xint, yint, k = 5)
        #
        return yint2
    def dosph(self, n, x, f, tilt = 1.5):
        #Function to do bessel integral using FFTLog for kernels
        f = f*self.renorm
        return sph(x, nu = n, q = tilt)(f, extrap = True)
    #PT kernels below
    #0 lag
    def xi0lin0(self, kmin = 1e-6, kmax = 1e3):
        val = quad(self.ilpk, kmin, kmax, limit = 200)[0]/self.tpi2
        return val
    #j0
    def xi0lin(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt)
    #j2
    def xi2lin(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(2, kint, integrand, tilt = tilt)
    #u1
    def u10lin(self, kint = None,  tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = -1*kint*self.ilpk(kint)
        integrand /= (kint**2.*self.tpi2)
        return self.dosph(1, kint, integrand, tilt = tilt)
    #correlatin function
    def corr(self, kint = None, tilt = 1.5):
        if kint is None:
            kint = self.kint
        integrand = self.ilpk(kint)
        integrand /= (1.*self.tpi2)
        return self.dosph(0, kint, integrand, tilt = tilt)
    #################
    #Bessel Integrals for \mu
    def template(self, k, l, func, expon, expon0, suppress, j0 = True):
        '''Generic template that is followed by mu integrals
        j0 is different since its exponent has sigma subtracted that is
        later used to suppress integral
        '''
        Fq = numpy.zeros_like(self.qv)
        if l:
            Fq = expon*func*self.yq**l
        elif j0:
            Fq = expon0*func
            
            
        ktemp, ftemp = \
            sph(self.qv, nu= l, q=max(0, 1.5 - l))(Fq*self.renorm,\
                extrap = False)
        if l==0 and j0:
            ftemp *= suppress
        return 1* k**l * numpy.interp(k, ktemp, ftemp)
    def integrals(self, k):
        '''Do the \mu integrals for various parameters for give 'k'
        '''
        expon = numpy.exp(-0.5*k**2 * self.XYlin)
        expon0 = numpy.exp(-0.5*k**2 * (self.XYlin - self.sigma))
        expon0m1 = numpy.expm1(-0.5*k**2 * (self.XYlin - self.sigma))
        suppress = numpy.exp(-0.5*k**2 *self.sigma)
        #
        za, b1, b2, b1sq, b2sq, b1b2 = 0, 0, 0, 0, 0, 0
        #l indep functions
        fza = 1.
        fb2sq = (self.corlin**2/2.)
        fb1 = (-2*self.qv*self.Ulin/self.Ylin)
        fb1b2 = (-2*self.qv*self.Ulin*self.corlin/self.Ylin)
        #
        for l in range(self.jn):
            #l-dep functions
            fb1sq = (self.corlin + (2*l/self.Ylin - k**2)*self.Ulin**2)
            fb2 = ((2*l/self.Ylin - k**2)*self.Ulin**2)
            #do integrals
            za   += self.template(k,l,fza,expon,expon0m1,suppress,j0=True)
            b1   += self.template(k,l,fb1,  expon,expon0,suppress,j0=False)
            b2   += self.template(k,l,fb2,  expon,expon0,suppress,j0=True)
            b1sq += self.template(k,l,fb1sq,expon,expon0,suppress,j0=True)
            b2sq += self.template(k,l,fb2sq,expon,expon0,suppress,j0=True)
            b1b2 += self.template(k,l,fb1b2,expon,expon0,suppress,j0=False)
        return 4*numpy.pi*numpy.array([za, b1, b2, b1sq, b2sq, b1b2])
    def make_table(self, kmin = 1e-3, kmax = 3, nk = 100):
        '''Make a table of different terms of P(k) between a given
        'kmin', 'kmax' and for 'nk' equally spaced values in log10 of k
        This is the most time consuming part of the code.
        '''
        self.pktable = numpy.zeros([nk, 8])
        kv = numpy.logspace(numpy.log10(kmin), numpy.log10(kmax), nk)
        self.pktable[:, 0] = kv[:]
        self.pktable[:, 1] = numpy.interp(kv, self.kp, self.p, left=0, right=0)
        for foo in range(nk):
            self.pktable[foo, 2:] = self.integrals(kv[foo])
    def extend_table(self,pklin,nk):
        """Inserts the linear theory P(k) into the array, given some user-specified number
        of sampling points nk."""
        # Insert linear theory.
        PL   = numpy.interp(self.pktable[:,0],pklin[:,0],pklin[:,1])
        self.pktable= numpy.insert(self.pktable,1,PL,axis=1)
        # Now extend the array to the highest k in Plin.
        ext      = numpy.zeros( (nk,self.pktable.shape[1]) )
        ext[:,0] = numpy.logspace(numpy.log10(self.pktable[-1,0]),numpy.log10(pklin[-1,0]),nk+1)[1:]
        ext[:,1] = numpy.interp(ext[:,0],pklin[:,0],pklin[:,1])
        self.pktable    = numpy.append(self.pktable,ext,axis=0)

    def write_table(self,fn):
        '''Writes the table to an ascii text file, fn.'''
        if self.pktable is None:
            print("Zeldovich table not created.")
            return
        # The order is k, ZA, b1, b2, b1sq, b2sq, b1b1
        hdr= ["k","P_L","P_Z","b1","b2","b1^2","b2^2","b1.b2"]
        ff = open(fn,"w")
        ff.write("# Components of the (real space) Zeldovich power spectrum.\n")
        str = "# %14s"%hdr[0]
        for hh in hdr[1:]:
            str += " %15s"%hh
        ff.write(str+"\n")
        for i in range(self.pktable.shape[0]):
            str = ""
            for t in self.pktable[i,:]:
                if numpy.isnan(t):
                    t = 0
                str += " %15.5e"%t
            ff.write(str+"\n")
        ff.close()
    def __call__(self,par,alphaZ=0,alphaL=0,sn=0):
        '''   
        A convenience function for the most common calls.
        par gives the coefficients for [k,P_lin,P_Z,b1,b2,b1^2,b2^2,b1.b2].
        For auto they are [0,0,1.,b1,b2,b1**2,b2**2,b1*b2],
        for cross they are [0,0,1.,b1/2.,b2/2.,0,0,0].
        Returns either the auto-spectrum or the halo-matter cross-spectrum
        depending upon auto.  This handles both the HaloZeldovich and ZEFT
        approximations, depending upon the value of alpha and sn.
        For ZEFT, you have two options for the counter-term, either use
        PZ (set by passing nonzero value to alphaZ) or use Plin
        (pass nonzero value to alphaL).
        '''
        if self.pktable is None:
            print("Zeldovich table not created.")
            return( (numpy.zeros(10),numpy.zeros(10)) )
        tmp = numpy.dot(self.pktable,par)
        # The EFT parameter, alpha, for the ZEFT model.
        tmp-= alphaL/2.*self.pktable[:,0]**2*self.pktable[:,1]
        tmp-= alphaZ/2.*self.pktable[:,0]**2*self.pktable[:,2]
        # The 1-halo parameter, sn, for the HaloZeldovich model.
        tmp+= sn
        return( (self.pktable[:,0],tmp) )
    def cross_spectrum(self,b1A=0,b2A=0,alphaA=0,snA=0,\
                            b1B=0,b2B=0,alphaB=0,snB=0):
        '''   
        The most general case, a cross-spectrum between two tracers
        with different bias, EFT and shot-noise parameters.
        CURRENTLY A STUB.
        '''
        return(0)