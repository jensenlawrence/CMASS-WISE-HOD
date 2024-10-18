import numpy as np
from astropy.cosmology import FlatLambdaCDM
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy import optimize

# Cosmology
p18_cosmo = FlatLambdaCDM(H0=67.66000,Om0=0.30964,Ob0=0.04897,Tcmb0=2.7255,Neff=3.046)
# with correct neutrino masses...
#p18_cosmo = FlatLambdaCDM(H0=67.66000,Om0=0.31105,Ob0=0.04897,Tcmb0=2.7255,Neff=3.046)

# Some settings you might want to play with
smin_ind = 7

# Some dndz-specific settings
estimator = 'ak'
error     = 'jackknife'
method    = 'loo_equalized'
Smin  = 0.1006799
Smax  = 10
nbins = 10
zmin_all = 0.00
zmax_all = 4.00
dz = 0.05

# Derived dndz-settings
bins = np.logspace(np.log10(Smin),np.log10(Smax),nbins+1)
bins_min = bins[:-1]
bins_max = bins[1:]
bins_avg = 0.5 * (bins_min + bins_max)
ds = bins_max - bins_min


# Directories

# Stuff that I need for the spec-z sample
# Directory where I measure magbias for the spectroscopic samples
magdir       = '../specz_magbias/'
# Directory where I put the spectroscopic dndz
speczdistdir = '../specz_zdist/'
# Directory where I put the specz bias measurements
speczbiasdir = '../specz_bias/'

# This specifies the dndz with a given scale cut.
# Here I am using 2.52 h^-1 Mpc but this could change...
bdndz_dir   = '../data/bdndz/bdndz-0.10-smin%.2f' % bins[smin_ind]

# xcorr directory
xcorr_dir  = '../data/xcorr_out-0.10/'

# Directory for xmatch dndz
xmatchdir    = '../dndz/xmatch_z/'
# Directory for sampled xmatch dn/dz
xmatch_dndz_sample_dir = '../dndz/xmatch_z_sampled/'

# Directory for magbias subtraction
magbias_subdir = '../dndz/magbias_subtract-0.10/smin-%.2f/' % bins_min[smin_ind]
# Directory for figures
figures_subdir = '../dndz/figures-0.10/smin-%.2f/' % bins_min[smin_ind]
# Directory for combined dn/dz
combodir = '../dndz/xcorr_z-0.10_measured_bias_cmass+lowz_all_err/'
# Directory for sampled dn/dz
dndz_sample_dir = '../dndz/xcorr_z_sampled-0.10_measured_bias_cmass+lowz_all_err_cut_off_z_gt_1.5/'

# Directory for bandpowers
bp_dir = '../bandpowers/OUTPUT_v10_ell6000_filter20_correctSN/'
# Directory for fitting
dndz_fit_dir = '../cell_fit/halofit_v10-0.10-measured_cmass+lowz_all_err_filter20_correct_SN_z_gt_3/'
# Directory for precomputed integrals
integral_dir = '../cell_integrals/'
# Directory for HSC xmatch
hsc_dir = '../dndz/xmatch_z_demp/'

# Galaxy colors to consider (these are part of the file names)
#colors    = ['blue','green','red','red_16.6','red_16.5','red_16.2']
colors    = ['blue','green','red_16.2']
#colors = ['green']
specs     = ['unwise_DR12_cmass','unwise_DR12_lowz','unwise_DR14','unwise_DR12QSO']
#specs = ['unwise_DR12_cmass','unwise_DR12_lowz','unwise_DR7','unwise_DR14','unwise_DR12QSO']
#specs = ['unwise_DR14']
#colors = ['red_16.2']
#specs = ['unwise_DR12_cmass']


def growth_factor(cc,zz):
	'''Retunrs linear growth factor for vector zz'''
	if isinstance(zz,float):
		zz = np.array([zz])
	afid = 1.0/(1.0+zz)
	if isinstance(afid,float):
		afid = np.array([afid])
	zval = 1./np.array(list(map(lambda x: np.logspace(x,0.0,100),np.log10(afid)))).transpose() - 1.0
	#zval = 1/np.logspace(np.log10(afid),0.0,100)-1.0
	Dz   = np.exp(-np.trapz(cc.Om(zval)**0.55,x=np.log(1/(1+zval)),axis=0))
	return(Dz)

def tracer_bias(z,option):
	'''Returns b(z) for a spectroscopic tracer'''
	if option == 'laurent-qso':
		#Quasar bias following Laurent+17
		return 0.278*((1+z)**2. - 6.565) + 2.393
	elif option == 'passive-cmass-white':
		# From White et al. 2011 (passive evolution model)
		b0 = 1.920
		z0 = 0.59
		return (b0-1.) * growth_factor(p18_cosmo,z0)/growth_factor(p18_cosmo,z) + 1.
	elif option == 'passive-lowz-parejko':
		# From Parejko et al. 2012 (Passive evolution model)
		b0 = 1.933
		z0 = 0.32
		return (b0-1.) * growth_factor(p18_cosmo,z0)/growth_factor(p18_cosmo,z) + 1.
	elif option == 'chiang-menard-cmass':
		# Digitized plot from Chiang & Menard 2019
		pass
	elif option == 'chiang-menard-lowz':
		# Digitized plot from Chiang & Menard 2019
		pass
	elif option == 'ak-cmass':
		# From my own measurements of b(z) from autocorrelation of CMASS
		data = np.loadtxt(speczbiasdir + 'cmass_auto_bias.txt')
		def biasfun(z):
			biasfun_orig = np.interp(z, 0.5 * (data[:,0] + data[:,1]), data[:,2])
			# Some cleanup of these outputs.
			# The z = 0.00 - 0.05 bin has a huge errorbar and should be excluded
			# so if z < 0.075, use the measurement from the 0.05 - 0.10 bin
			if z < 0.075:
				return data[1,2]
			elif (z >= 0.075 and z <= 0.775):
				return biasfun_orig
			elif z > 0.775:
				return 0
			# no data above z > 0.775, so just use the 
	elif option == 'ak-lowz':
		# From my own measurements of b(z) from autocorrelation of LOWZ
		data = np.loadtxt(speczbiasdir + 'lowz_auto_bias.txt')
		def biasfun(z):
			biasfun_orig = np.interp(z, 0.5 * (data[:,0] + data[:,1]), data[:,2])
			# Some cleanup of the output
			# The z=0.5 bias doesn't make sense (b=0.5), so at higher redshifts just use the z=0.475 result
			# Note that we should always cut off LOWZ at these high redshifts anyway
			if z > 0.475:
				return 0
			else:
				return biasfun_orig
	elif option == 'ak-eboss':
		# From my own measurements of b(z) from autocorrelation of eBOSS
		data = np.loadtxt(speczbiasdir + 'eboss_auto_bias_dz0.20.txt')
		def biasfun(z):
			biasfun_orig = np.interp(z, 0.5 * (data[:,0] + data[:,1]), data[:,2])
			# Some cleanup of the output
			# The error bars get very large below z=0.8 and above z=2.5, so just use the Laurent
			# fitting formula here
			# (note that Laurent seems to follow eBOSS quite well, in any event)
			if z < 0.7:
				return 0.278*((1+z)**2. - 6.565) + 2.393
			elif (z >= 0.7) & (z <= 2.3):
				return biasfun_orig	
			elif z > 2.3:
				return 0.278*((1+z)**2. - 6.565) + 2.393
	elif option == 'ak-boss':
		# From my own measurements of b(z) from autocorrelation of eBOSS
		data = np.loadtxt(speczbiasdir + 'boss_auto_bias_dz0.20.txt')
		def biasfun(z):
			biasfun_orig = np.interp(z, 0.5 * (data[:,0] + data[:,1]), data[:,2])
			# Some cleanup of the output
			# We can measure BOSS bias between 0.725 and 0.825; otherwise use Lauren
			# (note however that BOSS at the low-redshift range is somewhat above what
			# we would predict from Laurent)
			if z < 2.1:
				return 0.278*((1+z)**2. - 6.565) + 2.393
			elif (z >= 2.1) & (z <= 3.5):
				return biasfun_orig	
			elif (z > 3.5):
				return 0.278*((1+z)**2. - 6.565) + 2.393
	elif option == 'ak-dr7':
		data = np.loadtxt(speczbiasdir + 'sdssdr7_auto_bias.txt')
		def biasfun(z):
			biasfun_orig = np.interp(z, 0.5 * (data[:,0] + data[:,1]), data[:,2])
			# Some cleanup of the output
			# Note that again DR7 is above the expectation from Laurent...
			# (although the trend is pretty similar)
			# TBH I do not think my autocorrelation measurement of DR7 is particularly robust...
			if z < 0.3:
				return 0.278*((1+z)**2. - 6.565) + 2.393
			elif (z >= 0.3) & (z <= 0.5):
				return biasfun_orig	
			elif (z > 0.5) & (z < 0.9):
				return 0.278*((1+z)**2. - 6.565) + 2.393
			elif (z >= 0.9) & (z <= 1.9):
				return biasfun_orig
			elif (z > 1.9):
				return 0.278*((1+z)**2. - 6.565) + 2.393
	return map(biasfun, z)
	
def tracer_magbias_slope(tracer,zmin,zmax):
	'''Returns the magbias slope s for a given tracer.
	Measures s by redoing each sample selection if the fluxes
	are all lowered by 0.1 mags.  See details
	in referenced python codes'''
	if tracer == 'unwise_DR12_cmass':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_cmass.py
		sfile = np.loadtxt(magdir + 's_cmass_0.05.txt')
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,2]
		#return s[(zmin <= z) & (zmax >= z)]
		sinterp = np.interp(0.5*(zmin+zmax),zc,s)
		return sinterp
	elif tracer == 'unwise_DR12_lowz':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_lowz.py
		sfile = np.loadtxt(magdir + 's_lowz_0.05.txt')
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,2]
		#return s[(zmin <= z) & (zmax >= z)]
		sinterp = np.interp(0.5*(zmin+zmax),zc,s)
		return sinterp
	elif tracer == 'unwise_DR14':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_dr14.py
		sfile = np.loadtxt(magdir + 's_dr14.txt')
		# Note that because of the small # of quasars, I only use a few redshift bins
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,2]
		# Do a linear interpolation of the measured slope
		# with sensible extrapolation (using the extreme values)
		sinterp = np.interp(0.5*(zmin+zmax),zc,s,left=s[0],right=s[-1])
		return sinterp
	elif tracer == 'unwise_DR7':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_dr7.py
		# 0.7, constant with redshift looks about right
		# could be improved!
		sfile = np.loadtxt(magdir + 's_dr7.txt')
		# Note that because of the small # of quasars, I only use a few redshift bins
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,2]
		# Do a linear interpolation of the measured slope
		# with sensible extrapolation (using the extreme values)
		sinterp = np.interp(0.5*(zmin+zmax),zc,s,left=s[0],right=s[-1])
		return sinterp
	elif tracer == 'unwise_DR12QSO':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_dr12qso.py
		sfile = np.loadtxt(magdir + 's_dr12qso.txt')
		# Note that because of the small # of quasars, I only use a few redshift bins
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,2]
		# Do a linear interpolation of the measured slope
		# with sensible extrapolation (using the extreme values)
		sinterp = np.interp(0.5*(zmin+zmax),zc,s,left=s[0],right=s[-1])
		return sinterp
		
def tracer_magbias_slope_err(tracer,zmin,zmax):
	'''Returns the error in magbias slope s for a given tracer.
	Measures s by redoing each sample selection if the fluxes
	are all lowered by 0.1 mags.  See details
	in referenced python codes'''
	if tracer == 'unwise_DR12_cmass':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_cmass.py
		sfile = np.loadtxt(magdir + 's_cmass_0.05.txt')
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,3]
		#return s[(zmin <= z) & (zmax >= z)]
		sinterp = np.interp(0.5*(zmin+zmax),zc,s)
		return sinterp
	elif tracer == 'unwise_DR12_lowz':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_lowz.py
		sfile = np.loadtxt(magdir + 's_lowz_0.05.txt')
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,3]
		#return s[(zmin <= z) & (zmax >= z)]
		sinterp = np.interp(0.5*(zmin+zmax),zc,s)
		return sinterp
	elif tracer == 'unwise_DR14':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_dr14.py
		sfile = np.loadtxt(magdir + 's_dr14.txt')
		# Note that because of the small # of quasars, I only use a few redshift bins
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,3]
		# Do a linear interpolation of the measured slope
		# with sensible extrapolation (using the extreme values)
		sinterp = np.interp(0.5*(zmin+zmax),zc,s,left=s[0],right=s[-1])
		return sinterp
	elif tracer == 'unwise_DR7':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_dr7.py
		# 0.7, constant with redshift looks about right
		# could be improved!
		sfile = np.loadtxt(magdir + 's_dr7.txt')
		# Note that because of the small # of quasars, I only use a few redshift bins
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,3]
		# Do a linear interpolation of the measured slope
		# with sensible extrapolation (using the extreme values)
		sinterp = np.interp(0.5*(zmin+zmax),zc,s,left=s[0],right=s[-1])
		return sinterp
	elif tracer == 'unwise_DR12QSO':
		# Measurement comes from WISE_cross_CMB/cmb_lss_xcorr/magbias/measure_s_dr12qso.py
		sfile = np.loadtxt(magdir + 's_dr12qso.txt')
		# Note that because of the small # of quasars, I only use a few redshift bins
		zc = 0.5 * (sfile[:,0]+sfile[:,1])
		s    = sfile[:,3]
		# Do a linear interpolation of the measured slope
		# with sensible extrapolation (using the extreme values)
		sinterp = np.interp(0.5*(zmin+zmax),zc,s,left=s[0],right=s[-1])
		return sinterp
			
def tracer_dndz(tracer,zmin,zmax):
	'''Get dndz of the tracer samples.'''
	# from specz_dist/make_specz_zdist.py
	zbin = np.linspace(0,4,401)
	zc = 0.5 * (zbin[1:]+zbin[:-1])
	if tracer == 'unwise_DR7':
		zdist = np.loadtxt(speczdistdir + 'dr7qso.txt')
	elif tracer == 'unwise_DR12_cmass':
		zdist = np.loadtxt(speczdistdir + 'dr12cmassN.txt')
	elif tracer == 'unwise_DR12_lowz':
		zdist = np.loadtxt(speczdistdir + 'dr12lowzN.txt')
	elif tracer == 'unwise_DR14':
		zdist = np.loadtxt(speczdistdir + 'dr14qsoN.txt')
	elif tracer == 'unwise_DR12QSO':
		zdist = np.loadtxt(speczdistdir + 'dr12qsoN.txt')
	zdist[zc < zmin] = 0
	zdist[zc >= zmax] = 0
	return np.array([zc,zdist/np.sum(zdist)]).transpose()
	
def wise_bias(z,color):
	'''A rough fit to the wise b(z). To see that this works,
	run implied_bias_rmin2.52_propagate_dndz_error.py and overplot 
	the function from here'''
	if color == 'blue':
		return 0.8 + 1.2 * z
	elif color == 'green':
		return np.max([1.6 * z ** 2.0, 1])	
	elif color == 'red' or color == 'red_16.6' or color == 'red_16.5' or color == 'red_16.2':
		return np.max([2.0 * z ** 1.5, 1])

def wise_magbias_slope(color):
	# From number_density_slope_w1_w2.py and number_density_slope.py
	if color == 'red':
		return  0.695
	elif color == 'green':
		return  0.648
	elif color == 'blue':
		return  0.455
	elif color == 'red_16.6':
		return  0.744
	elif color == 'red_16.5':
		return  0.773
	elif color == 'red_16.2':
		return  0.842
		
def wise_magbias_slope_err(color):
	# equal to 0.5 the range between s for lambda_min = 0 and lambda_min = 80
	if color == 'red':
		return 0.018
	elif color == 'green':
		return 0.048
	elif color == 'blue':
		return 0.026
	elif color == 'red_16.6':
		return 0.011
	elif color == 'red_16.5':
		return 0.011
	elif color == 'red_16.2':
		return 0.026
		
def xmatch_dndz(color,measured=True):
	# Load the redshift distributions
	if measured:
		dat = np.loadtxt(xmatchdir + color + '.txt')
	else:
		dat = np.loadtxt(xmatch_dndz_sample_dir + color + '.txt')
	z = dat[:,0]
	dndzi = dat[:,1]
	dndzi = dndzi/np.sum(dndzi*np.gradient(z))
	return np.array([z,dndzi]).transpose()
	
def xcorr_dndz(color,splineflag):
	#def xcorr_dndz(color):
	dat = np.loadtxt(dndz_sample_dir + splineflag + '/' + color + '.txt')
	#dat = np.loadtxt(dndz_sample_dir + '/' + color + '.txt')
	z = dat[:,0]
	fit = dat[:,1]
	fit = fit/np.trapz(fit,x=z)
	return  np.array([z,fit]).transpose()
	
def wbar_amplitude(cc,zz,pk,gamma,nonlinear=False):
	'''Takes care of the
	\int \frac{ k \ dk}{2 \pi} J_0(k r) P(k, z) \int_{r_{\rm min}}^{r_{\rm max}} r^{\gamma} dr
	in \bar{w}.
	Either you are in the linear regime, in which case you can just pull out D^2(z)
	and don't worry about the integral (since it's a redshift-independent normalization
	that you will just divide out anyway) or you are in the nonlinear regime and you need to compute the integral
	at each redshift.'''
	if nonlinear==False:
		return(growth_factor(cc,zz)**2.)
	else:
		pars = {'method': 'halofit','tracer1': {'b': lambda z: 1.0}, 'tracer2': {'b': lambda z: 1.0}}
		rmin = 2.5
		rmax = 10.0
		kmin = 0.0
		kmax = 10.0
		dk = 0.1
		nk = (kmax-kmin)/dk + 1
		kval = np.linspace(kmin, kmax, nk)
		p_mm = np.array(list(map(lambda z: pk(pars, kval, z)[3], zz)))
		#p_mm = np.zeros_like(zz)
		#for i,z in enumerate(zz):
		#	p_mm[i] = pk(pars, kval, z)[3]

		bin_integral = 2./(bins_max[smin_ind:]**2. - bins_min[smin_ind:]**2.) * ((bins_max[smin_ind:] * special.j1(kval[:,np.newaxis] * bins_max[smin_ind:]) / kval[:,np.newaxis]) - (bins_min[smin_ind:] * special.j1(kval[:,np.newaxis] * bins_min[smin_ind:])/kval[:,np.newaxis]))
		bin_tot = np.sum(bin_integral * bins_avg[smin_ind:]**gamma * (bins_max[smin_ind:]-bins_min[smin_ind:]),axis=1)

		return (dk/(2*np.pi)) * np.nansum(p_mm * bin_tot[np.newaxis,:] * kval[np.newaxis,:], axis=1)

def wbar_amplitude_full(cc,zz,dndz_spec,pk,gamma):
	'''Takes care of the
	\int \frac{ k \ dk}{2 \pi} J_0(k r) P(k, z) \int_{r_{\rm min}}^{r_{\rm max}} r^{\gamma} dr
	in \bar{w}.
	Either you are in the linear regime, in which case you can just pull out D^2(z)
	and don't worry about the integral (since it's a redshift-independent normalization
	that you will just divide out anyway) or you are in the nonlinear regime and you need to compute the integral
	at each redshift.'''	
	pars = {'method': 'halofit','tracer1': {'b': lambda z: 1.0}, 'tracer2': {'b': lambda z: 1.0}}
	rmin = 2.5
	rmax = 10.0
	kmin = 0.0
	kmax = 10.0
	dk = 0.1
	nk = (kmax-kmin)/dk + 1
	kval = np.linspace(kmin, kmax, nk)
	p_mm = np.zeros((int(nk),len(zz)))
	for i,z in enumerate(zz):
		p_mm[:,i] = pk(pars, kval, z)[3]
	
	dz = zz[1]-zz[0]
	
	area = np.pi * (bins_max[smin_ind:]**2. - bins_min[smin_ind:]**2.)
	prefactor = bins_avg[smin_ind:]**gamma * (bins_max[smin_ind:]-bins_min[smin_ind:]) / area
	
	bin_tot = 0
	for i in range(len(area)):
		rmax = bins_max[smin_ind:][i]
		rmin = bins_min[smin_ind:][i]
		
		first_term = rmax * special.j1(kval * rmax) / kval
		second_term = rmin * special.j1(kval * rmin) / kval
		
		bin_integral = prefactor[i] * dz * dk * np.nansum(((first_term - second_term) * kval)[:,np.newaxis] * dndz_spec * p_mm)
		#bin_integral = prefactor[i] * dk * np.nansum(((first_term - second_term) * kval) * p_mm[:,1])
		#bin_integral = prefactor[i] * dk * (first_term - second_term)
		
		bin_tot += bin_integral

	#I2 = np.nansum(bin_tot * kval * p_mm[:,1])

	return bin_tot

def get_zatchi_chival(cosmo, Nchi, zmin, zmax):
	'''Split out the zatchi/chival calculation for dndz norm, since
	it is rather time-consuming.'''
	hub      = cosmo.H0.value / 100.
	zval     = np.linspace(zmin,zmax,1000) # Accuracy doesn't depend much on the number of zbins
	chival   = p18_cosmo.comoving_distance(zval).value*hub	# In Mpc/h.
	zatchi   = Spline(chival,zval)
	# Spline comoving distance as well to make it faster
	chiatz   = Spline(zval, chival)
	# Work out W(chi) for the objects whose dNdz is supplied.
	chimin   = np.min(chival) + 1e-5
	chimax   = np.max(chival)
	chival   = np.linspace(chimin,chimax,Nchi)
	zval     = zatchi(chival)
	return zatchi, chival
	
# Order of the spline coefficients
k = 3
# Global settings for dndz, used for precompute_integrals.py
zmin_prec = 0.0
zmax_prec = 4.0
nz = 401
z = np.linspace(zmin_prec,zmax_prec,nz)
zmax = 3.0

zatchi, chival = get_zatchi_chival(p18_cosmo, 500,zmin_prec,zmax_prec)

def normalize_dndz(cosmo, dndz, zatchi, chival):	
	'''Function to normalize dndz.  Needed to correct the basis functions for the normalization.'''		
	fchi1    = np.interp(zatchi(chival),\
		dndz[:,0],dndz[:,1]*cosmo.H(dndz[:,0]).value,left=0,right=0)
	return np.trapz(fchi1,x=chival)

def b2_from_b1_PS(b1_lagrangian):
	'''Eqn. 2.9 in Modi Vlah and White.
	Relates the Lagrangian b1 and b2 in peak-background split with
	spherical collapse (Press-Schechter mass function).'''
	deltac = 1.69
	nu = np.sqrt((deltac * b1_lagrangian) + 1)
	return (nu**4. - 3 * nu**2.)/deltac**2.
	
def b2_from_b1_SMT(b1_lagrangian):
	'''Same as above but for Sheth-Mo-Tormen mass function.
	Derived from eqns. 3.31-3.32 and 2.16 in https://arxiv.org/pdf/1611.09787.pdf'''
	deltac = 1.69
	p = 0.3
	q = 0.707
	if type(b1_lagrangian) != np.ndarray:
		b1_lagrangian = np.array([b1_lagrangian])
	nus = np.zeros_like(b1_lagrangian)
	for i,b1L in enumerate(b1_lagrangian):
		def f(nu):
			return (q * nu**2. - 1.)/deltac + ( (2. * p)/deltac)/(1. + (q * nu**2.)**p) - b1L
		nu = optimize.newton(f, 1.0)
		nus[i] = nu
	return (q * nus**2./deltac**2.) * (q * nus**2. - 3.) + (2 * p/deltac**2.) * (-1. + 2. * p + 2. * q * nus**2.) * 1./(1. + (q * nus**2.)**p)
	
def b2_from_b1_lazeyras(b1_lagrangian):
	'''From fitting function in Lazeyras+16, eqn. 5.2 https://arxiv.org/pdf/1511.01096.pdf
	This is a fitting function to Eulerian b1/b2 so it needs to also be adjusted to lagrangian b2
	using b1E = 1 + b1L
	and b2E = 8/21 b1L + b2L'''
	return (-8./21.) * b1_lagrangian + 0.412 - 2.143 * (1. + b1_lagrangian) + 0.929 * (1 + b1_lagrangian)**2. + 0.008 * (1 + b1_lagrangian)**3.
	
def b2_from_b1_modi(b1_lagrangian):
	'''From Fig. 9 in Modi, Castorina and Seljak + 16 which directly gives b2_lagrangian as a function of b1_lagrangian.
	I had to digitize the plot because the fit isn't available in the paper.'''
	dat = np.loadtxt('/Users/ALEX/Berkeley/WISE_cross_CMB/cmb_lss_xcorr/zeldovich/modi_bias.csv',delimiter=',')
	s = Spline(dat[:,0],dat[:,1],k=2,ext='raise')
	return s(b1_lagrangian)
	
def wise_constant_bias(z,color,cross=True):
	'''The best-fit WISE b_eff from either the cross or the auto.'''
	if cross == True:
		if color == 'blue':
			return 1.56
		elif color == 'green':
			return 2.25
		elif color == 'red_16.2':
			return 3.49
	elif cross == False:
		if color == 'blue':
			return 1.71
		elif color == 'green':
			return 2.46
		elif color == 'red_16.2':
			return 3.29