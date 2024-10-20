#!/usr/bin/env python3
#
# Python code to compute the Limber integral to convert
# P(k) -> C_l.
#

import numpy as np
import astropy.cosmology
from   scipy.interpolate import InterpolatedUnivariateSpline as Spline
import glob
import sys
import re
import time
import os


import zeldovich as Z

#def znorm(dndz):
#	'''return dn/dz normalized in redshift'''
#	return dndz[:,1]/np.trapz(dndz[:,1],x=dndz[:,0])

class PowerSpectrum():
	"""A class which returns P(k) from pre-computed power spectrum
	information."""
	# The main purpose of this class is to smoothly handle the
	# interpolation in z.  If we have a small-ish number of points
	# Lagrange interpolating polynomials are ideal for vectorized ops.
	# Currently P(k) is computed from external files assumed to be
	# pre-generated by "generate_PT_tables".
	def lagrange_spam(self,z):
		"""Returns the weights to apply to each z-slice to interpolate to z."""
		dz = self.zlist[:,None] - self.zlist[None,:]
		singular = (dz == 0)
		dz[singular] = 1.0
		fac = (z - self.zlist) / dz
		fac[singular] = 1.0
		return(fac.prod(axis=-1))
	def __init__(self):
		"""Initialize the class by reading the pre-generated P(k) tables.
		Read both PT tables and HF tables."""
		if sys.path[0]:
			dirpath = sys.path[0] + '/'
		else:
			dirpath = sys.path[0]
		flist=sorted(glob.glob(dirpath + "HF_tables/HF_table_z???.txt"))
		print(dirpath + "HF_tables/HF_table_")
		rex  =re.compile(dirpath + "HF_tables/HF_table_z(\d\d\d).txt")
		self.pk    = None
		self.zlist = []
		self.zeldo = []
		self.halofit = []
		for fn in flist:
			mm = rex.search(fn)
			if mm!=None:
				self.zlist.append(float(mm.group(1))/100.)
				# Note all that matters for Zeldovich is the pktable,
				# so just initialize it however we want.
				thiszel = Z.Zeldovich(np.arange(10),np.arange(10))
				table = np.loadtxt(fn)
				thiszel.pktable  = table
				self.zeldo.append(thiszel)       
				self.halofit.append(np.loadtxt(dirpath + "HF_tables/HF_table_z{}.txt".format(mm.group(1))))
		self.zlist = np.array(self.zlist)
	def __call__(self,pars,kval,zz):
		"""Returns power spectra at k=kval and z=zz.
		pars is a dict telling you which method to use (halozeldovich, zeft, halofit)
		and the bias/1-halo parameters for 2 tracers.  
		ex: pars = {'method': 'halofit', 'tracer1': {'b': lambda z: 1.5},
		'tracer2': {'b': lambda z: 2.0}}.
		Note that biases must be functions of redshift.
		This function then returns all 4 spectra needed for CMB lensing:
		tracer1 x tracer2 , tracer1 x matter, tracer2 x matter, and matter x matter.
		If you want auto-spectra, just use the same bias coefficients for tracer1 and tracer2."""
		# Get the interpolating coefficients then just weigh each P(k,z).
		coef = self.lagrange_spam(zz)
		p_auto = np.zeros_like(kval)
		p_cross1 = np.zeros_like(kval)
		p_cross2 = np.zeros_like(kval)
		p_mat = np.zeros_like(kval)
		for c,z,h in zip(coef,self.zeldo,self.halofit):
			if pars['method'] == 'linear':
				b1 = pars['tracer1']['b'](zz)
				b2 = pars['tracer2']['b'](zz)
				
				par = np.array([0,b1*b2,0,0,0,0,0,0])
				kk,pk = z(par)
				p_auto += np.interp(kval,kk,pk,left=0,right=0) * c
				
				par = np.array([0,b1,0,0,0,0,0,0])
				kk,pk = z(par)
				p_cross1 += np.interp(kval,kk,pk,left=0,right=0) * c				

				par = np.array([0,b2,0,0,0,0,0,0])
				kk,pk = z(par)
				p_cross2 += np.interp(kval,kk,pk,left=0,right=0) * c				

				par = np.array([0,1,0,0,0,0,0,0])
				kk,pk = z(par)
				p_mat += np.interp(kval,kk,pk,left=0,right=0) * c				
				
			if (pars['method'] == 'halozeldovich') or (pars['method'] == 'zeft') or (pars['method'] == 'zeft-lin'):        		
				b1 = 0.5 * (pars['tracer1']['b1'](zz) + pars['tracer2']['b1'](zz))
				b2 = 0.5 * (pars['tracer1']['b2'](zz) + pars['tracer2']['b2'](zz))
				b1sq = pars['tracer1']['b1'](zz)*pars['tracer2']['b1'](zz)
				b2sq = pars['tracer1']['b2'](zz)*pars['tracer2']['b2'](zz)
				b1b2 = 0.5 * (pars['tracer1']['b1'](zz)*pars['tracer2']['b2'](zz) + pars['tracer1']['b2'](zz)*pars['tracer2']['b1'](zz))
				par = np.array([0,0,1.,b1,b2,b1sq,b2sq,b1b2])

				if pars['method'] == 'halozeldovich':
					kk,pk = z(par,sn=pars['sn'](zz))
				elif pars['method'] == 'zeft':
					kk,pk = z(par,alphaZ=pars['alpha'](zz))  	
				elif pars['method'] == 'zeft-lin':
					kk,pk = z(par,alphaL=pars['alpha'](zz))  	
				p_auto += np.interp(kval,kk,pk,left=0,right=0) * c

				b1 = pars['tracer1']['b1'](zz)
				b2 = pars['tracer1']['b2'](zz)
				par = np.array([0,0,1.,b1/2.,b2/2.,0,0,0])
				if pars['method'] == 'halozeldovich':
					kk,pk = z(par,sn=pars['sn'](zz))
				elif pars['method'] == 'zeft':
					kk,pk = z(par,alphaZ=pars['alpha'](zz))
				elif pars['method'] == 'zeft-lin':
					kk,pk = z(par,alphaL=pars['alpha'](zz))
				p_cross1 += np.interp(kval,kk,pk,left=0,right=0) * c 

				b1 = pars['tracer2']['b1'](zz)
				b2 = pars['tracer2']['b2'](zz)
				par = np.array([0,0,1.,b1/2.,b2/2.,0,0,0])
				if pars['method'] == 'halozeldovich':
					kk,pk = z(par,sn=pars['sn'](zz))
				elif pars['method'] == 'zeft':
					kk,pk = z(par,alphaZ=pars['alpha'](zz))
				elif pars['method'] == 'zeft-lin':
					kk,pk = z(par,alphaL=pars['alpha'](zz))
				p_cross2 += np.interp(kval,kk,pk,left=0,right=0) * c  

				# Always use Halofit for the magnification bias term (P_MM)
				pk = h[:,1]    		
				p_mat += np.interp(kval,kk,pk,left=0,right=0) * c	      
			elif pars['method'] == 'halofit':
				kk = h[:,0]
				pk = h[:,1] * pars['tracer1']['b'](zz) * pars['tracer2']['b'](zz)      		
				p_auto += np.interp(kval,kk,pk,left=0,right=0) * c

				pk = h[:,1] * pars['tracer1']['b'](zz)     		
				p_cross1 += np.interp(kval,kk,pk,left=0,right=0) * c

				pk = h[:,1] * pars['tracer2']['b'](zz)        	
				#print 5/0
				p_cross2 += np.interp(kval,kk,pk,left=0,right=0) * c

				pk = h[:,1]    		
				p_mat += np.interp(kval,kk,pk,left=0,right=0) * c
		return(p_auto, p_cross1, p_cross2, p_mat)

def mag_bias_kernel(cosmo, dndz, s, zatchi, chival, chivalp, zvalp, Nchi_mag=1000):
	"""Returns magnification bias kernel as a function of chival/zval. Arguments
	'cosmo' is an astropy Cosmology instance,
	'dndz' is the redshift distribution,
	's' is the slope of the number counts dlog10N/dm,
	'chival' is the array of comoving distances for which the magnification
	bias kernel is defined (and 'zval' is the corresponding array of redshifts),
	'chistar' is the comoving distance to the surface of last scattering,
	'zatchi' is a function taking comoving distance as input and returning redshift,
	and 'Nchi_mag' gives the number of sampling points in the integral over chiprime.
	"""
	dndz_norm = np.interp(zvalp,\
						dndz[:,0],dndz[:,1],left=0,right=0)
	norm = np.trapz( dndz[:,1], x = dndz[:,0])
	#print('dndz',dndz[:,1])
	#print('norm',norm)
	#if norm == 0:
	#	norm = np.sum(dndz[:,1]) * (dndz[1,0]-dndz[0,0])
	#	dndz_norm  = dndz_norm/norm
	#	print('chivalp',chivalp)
	#	print(5/0)
	#	g = chival * np.sum( 1. / (chivalp) * (chivalp - chival[np.newaxis,:]) * dndz_norm * (1.0/2997.925) * cosmo.H(zvalp).value/cosmo.H(0).value, x =  chivalp, axis=0)
	#else:
	dndz_norm  = dndz_norm/norm
	g = chival * np.trapz( 1. / (chivalp) * (chivalp - chival[np.newaxis,:]) * dndz_norm * (1.0/2997.925) * cosmo.H(zvalp).value/cosmo.H(0).value, x =  chivalp, axis=0)
	
	mag_kern = 1.5 * (cosmo.Om0) * (1.0/2997.925)**2 * (1+zatchi(chival)) * g * (5. * s - 2.)
	return mag_kern

def setup_chi(cosmo, dndz1, dndz2, Nchi, Nchi_mag,zmin=None, zmax=None):
	hub      = cosmo.H0.value / 100.
	#if not zmin:
	zmin     = np.min(np.append(dndz1[:,0],dndz2[:,0]))
	#if not zmax:
	zmax     = np.max(np.append(dndz1[:,0],dndz2[:,0]))
	zval     = np.linspace(zmin,zmax,1000) # Accuracy doesn't depend much on the number of zbins
	chival   = cosmo.comoving_distance(zval).value*hub	# In Mpc/h.
	zatchi   = Spline(chival,zval)
	# Spline comoving distance as well to make it faster
	chiatz   = Spline(zval, chival)
	# Work out W(chi) for the objects whose dNdz is supplied.
	chimin   = np.min(chival) + 1e-5
	chimax   = np.max(chival)
	chival   = np.linspace(chimin,chimax,Nchi)
	zval     = zatchi(chival)
	chistar  = cosmo.comoving_distance(1098.).value * hub
	chivalp = np.array(list(map(lambda x: np.linspace(x,chistar,Nchi_mag),chival))).transpose()
	zvalp = zatchi(chivalp)
	return zatchi, chiatz, chival, zval, chivalp, zvalp
	
def do_limber(ell, cosmo, dndz1, dndz2, s1, s2, pk, pars, crossCMB=True, autoCMB=False, use_zeff=True, Nchi=50, mag_bias='all', dndz1_mag=None, dndz2_mag=None, normed=False, normed1=False, setup_chi_flag=False, setup_chi_out=None, normalize_cmb_kernel=False, zmin=None, zmax=None):
	"""Does the Limber integral, returning Cell.
	If use_zeff==True then P(k,z) is assumed to be P(k,zeff).
	On input:
	'ell' is an array of ells for which you want Cell.
	'cosmo' is an astropy Cosmology instance,
	'dndz1' contains two columns (z and dN/dz for sample 1),
	'dndz2' contains two columns (z and dN/dz for sample 2),
	's1' and 's2' are the slope of number counts dlog10N/dm for the 2 samples
	(for magnification bias),
	'pk' is a power spectrum instance (k and P, in Mpc/h units),
	'pars' is a dict with elements 'method' (which can be 'halozeldovich',
	'zeft', or 'halofit'), 'tracer1' giving the bias terms for tracer1
	(Lagrangian b1 and b2 for halozeldovich/zeft or Eulerian b for halofit),
	optionally tracer2, and sn/alpha for halozeldovich/zeft.  Biases, sn and alpha
	must be given as functions of redshift.
	If crossCMB is false returns the object-object auto-correlation while
	if crossCMB is true returns object-CMB(kappa) cross-correlation
	neglecting dndz2.
	'Nchi' gives the number of integration points,
	and 'mag_bias' can be 'all' (all terms), 'only' (mu-mu or mu-kappa only) or 'cross': (2 x mu-galaxy).
	(if you only want the clustering term, set s1=s2=0.4).
	'dndz1_mag' and 'dndz2_mag' are optional additional dn/dz to use in the magnification
	bias term (if you want to use a different dn/dz here and in the clustering term).

	Per test_limber.py, Nchi=1000 is accurate to 0.08% for ell < 1000 and 0.3% for ell < 2000.
	[this depends on how 'spiky' dn/dz is....]

	This is the ``simplified'' Limber code which takes as input P_{NN}, the power spectrum
	of the non-neutrino density fluctuations.  In CAMB-speak, this is P(k) with var1=delta_nonu
	and var2=delta_nonu.  See limber2_complex.py for an explanation of this approximation.
	Empirically this works to ~0.1% at ell > 100; see the limber2 and limber2_complex tests
	(and note how similar they are!)."""
	# Set up cosmological parameters.
	if 'tracer2' not in pars:
		pars['tracer2'] = pars['tracer1']
	# Set up the basic distance-redshift conversions.
	hub      = cosmo.H0.value / 100.
	Nchi_mag = 1000 # Number of sampling points in magnification bias integral
	if setup_chi_flag == False:	
		zatchi, chiatz, chival, zval, chivalp, zvalp = setup_chi(cosmo, dndz1, dndz2, Nchi, Nchi_mag, zmin, zmax)
	else:
		zatchi, chiatz, chival, zval, chivalp, zvalp = setup_chi_out
		#print 5/0
	
	fchi1    = np.interp(zatchi(chival),\
		 dndz1[:,0],dndz1[:,1]*cosmo.H(dndz1[:,0]).value,left=0,right=0)
	#print('fchi1',fchi1)
	if not normed:
		norm1 = np.copy(np.trapz(fchi1,x=chival))
		fchi1   /= norm1
	fchi2    = np.interp(zatchi(chival),\
		 dndz2[:,0],dndz2[:,1]*cosmo.H(dndz2[:,0]).value,left=0,right=0)
	if not normed:
		norm2 = np.copy(np.trapz(fchi2,x=chival))
		fchi2   /= norm2

	if dndz1_mag is None:
		mag_kern1 = mag_bias_kernel(cosmo, dndz1, s1, zatchi, chival, chivalp, zvalp, Nchi_mag)
	else:
		mag_kern1 = mag_bias_kernel(cosmo, dndz1_mag, s1, zatchi, chival, chivalp, zvalp, Nchi_mag)
	if dndz2_mag is None:
		mag_kern2 = mag_bias_kernel(cosmo, dndz2, s2, zatchi, chival, chivalp, zvalp, Nchi_mag)
	else:
		mag_kern2 = mag_bias_kernel(cosmo, dndz2_mag, s2, zatchi, chival, chivalp, zvalp, Nchi_mag)
	# If we're not doing galaxy-galaxy cross-corelations make
	# fchi2 be W(chi) for the CMB.
	
	chistar  = cosmo.comoving_distance(1098.).value * hub
	
	#np.savetxt('mag_kernel.txt',mag_kern1)
	#np.savetxt('zv.txt',zatchi(chival))
	#np.savetxt('dndz1_mag.txt',dndz1_mag)
	#print(5/0)

	if crossCMB:
		fchi2    = 1.5* (cosmo.Om0) *(1.0/2997.925)**2*(1+zatchi(chival))
		fchi2   *= chival*(chistar-chival)/chistar
		if normalize_cmb_kernel:
			fchi2   /= np.trapz(fchi2,x=chival)
	if autoCMB:
		fchi2    = 1.5* (cosmo.Om0) *(1.0/2997.925)**2*(1+zatchi(chival))
		fchi2   *= chival*(chistar-chival)/chistar
		fchi1 = fchi2
		
	if zmin is not None:
		#fchi1_spline = Spline(chival, fchi1)
		#fchi2_spline = Spline(chival, fchi2)
		chimin   = cosmo.comoving_distance(zmin).value*hub
		chimax   = cosmo.comoving_distance(zmax).value*hub
		chival   = np.linspace(chimin,chimax,Nchi)
		#fchi1 = fchi1_spline(chival)
		#fchi2 = fchi2_spline(chival)
		fchi1    = np.interp(zatchi(chival),\
			dndz1[:,0],dndz1[:,1]*cosmo.H(dndz1[:,0]).value,left=0,right=0)
		fchi2    = np.interp(zatchi(chival),\
			dndz2[:,0],dndz2[:,1]*cosmo.H(dndz2[:,0]).value,left=0,right=0)
		if not normed:
			fchi2 /= np.trapz(fchi2,x=chival)
		if not normed1:
			fchi1 /= np.trapz(fchi1,x=chival)
		elif normed1:
			fchi1 /= norm1
		

		# Get effective redshift     
	if use_zeff:
		kern = fchi1*fchi2/chival**2
		zeff = np.trapz(kern*zval,x=chival)/np.trapz(kern,x=chival)
		print("zeff=",zeff)
	else:
		zeff = -1.0

	#print fchi1
	#print fchi2
	# and finally do the Limber integral.
	np.savetxt('fchi1.txt',np.array([chival,fchi1]).T)
	np.savetxt('fchi2.txt',np.array([chival,fchi2]).T)
	Nell = len(ell)
	cell = np.zeros( (Nell, Nchi) )
	for i,chi in enumerate(chival):
		if (fchi2[i] != 0) | (mag_kern2[i] != 0):
			kval = (ell+0.5)/chi
			if use_zeff:
				# Assume a fixed P(k,zeff).
				# This should return both pofk for the galaxy [with the bias]
				# and pofk for matter [no bias, but would include shotnoise] 
				p_auto, p_cross1, p_cross2, p_mat = pk(pars, kval, zeff)
				#print p_auto, p_cross1, p_cross2, p_mat
			else:
				# Here we interpolate in z.
				zv   = zatchi(chi)
				p_auto, p_cross1, p_cross2, p_mat = pk(pars, kval, zv)
				#print chi, p_auto, p_cross1, p_cross2, p_mat
				#print p_auto, p_cross1, p_cross2, p_mat
			#print 'chi', chi
			#print 'p_auto', p_auto
			#print 'zv', zv
			#print 'kval', kval

				
			if not crossCMB:
				f1f2 = fchi1[i]*fchi2[i]/chi**2 * p_auto 
				f1m2 = 	fchi1[i]*mag_kern2[i]/chi**2 * p_cross1
				m1f2 = mag_kern1[i]*fchi2[i]/chi**2 *  p_cross2 
				m1m2 = mag_kern1[i]*mag_kern2[i]/chi**2 * p_mat
				#print f1f2, f1m2, m1f2, m1m2
				#print p_auto, p_cross1, p_cross2, p_mat				
				if mag_bias == 'only':
					cell[:,i] = m1m2
				elif mag_bias == 'cross1':
					cell[:,i] = f1m2
				elif mag_bias == 'cross2':
					cell[:,i] = m1f2
				elif mag_bias == 'all':
					cell[:,i] = f1f2 + f1m2 + m1f2 + m1m2
					#print('fchi1[i]',fchi1[i])
					#print('mag_kern2[i]',mag_kern2[i])
			else:
				f1f2 = fchi1[i]*fchi2[i]/chi**2 * p_cross1
				m1f2 = mag_kern1[i]*fchi2[i]/chi**2 * p_mat
				if mag_bias == 'only':
					cell[:,i] = m1f2
				elif mag_bias == 'all':
					cell[:,i] = m1f2 + f1f2
					#print fchi1[i], fchi2[i], chi
	cell = np.trapz(cell,x=chival,axis=-1)
	#print(5/0)
				#
	return( cell )
	#






if __name__=="__main__":
    if len(sys.argv)!=2:
        raise RuntimeError("Usage: {:s} ".format(sys.argv[0])+\
                           "<dndz-fname>")
    else:
        # Assume these are ascii text files with two columns each:
        # z,dNdz for the first and k,P(k) for the second.  The best
        # interface to be determined later.
        dndz1= np.loadtxt(sys.argv[1])
        dndz2= np.loadtxt(sys.argv[1])
        pk   = PowerSpectrum()
        l,Cl = do_limber(astropy.cosmology.Planck15,dndz1,dndz2,pk,\
                         auto=False,use_zeff=True)
        print(l,Cl)
        l,Cl = do_limber(astropy.cosmology.Planck15,dndz1,dndz2,pk,\
                         auto=True,use_zeff=False)
        print(l,Cl)
    #
