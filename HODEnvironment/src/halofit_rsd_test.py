import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck15
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
from scipy.interpolate import RectBivariateSpline
import sys
from scipy.special import legendre, spherical_jn
import astropy.units as u

import os
import time

import limber2 as L
from settings import *

# Adjustable Options
def w_theta(b1, b2, dndz, dndz_spec_all, zmin, zmax, dz, theta, normed1=False):
	#color = 'blue'
	#spec = 'unwise_DR12_cmass'
	#zmin = 0.45
	#zmax = 0.50
	#dz = 0.05
	#swise = 0.453

	# Cosmology
	p18_cosmo = FlatLambdaCDM(H0=67.74,Om0=0.3075,Ob0=0.0486,Tcmb0=2.7255,Neff=3.046)

	# Load the power spectrum
	pk = L.PowerSpectrum()

	# Load the redshift distributions
	#os.chdir('../')
	#dndz = xmatch_dndz(color)


	# lmax = 50000
	ell = np.linspace(0,5e4,int(5e4+1)) # 0.5% convergence compared to 1 point per ell


	data = np.loadtxt('data/combined_data.txt')
	R = data[:,0][7:]
	distance = 0.5 * (Planck15.comoving_distance(zmin) + Planck15.comoving_distance(zmax))


	# Set dndz for the spec sample
	# z_spec = np.linspace(0,4,1000)
	#dndz_spec_all = tracer_dndz(spec,zmin,zmax)

	pars = {'method': 'halofit','tracer1': {'b': lambda z: b1}, 'tracer2': {'b': lambda z: b2}}

	clgg = L.do_limber(ell, p18_cosmo, dndz, dndz_spec_all,  0.4, 0.4, pk, pars, crossCMB=False, use_zeff=False, Nchi = 500, zmin=0.45, zmax=0.5, normed1=normed1)
	np.savetxt('wchi_test.txt',np.array([ell,clgg]).T)
	dell = np.gradient(ell)
	w_theory = np.sum(dell/(2*np.pi)*ell *clgg * special.jv(0,ell*theta[:,np.newaxis]),axis=1)

	return w_theory

def xi(b1, b2, zmin, zmax, r):	
	sys.path[0] = '/project/6033532/jptlawre'
	pk = L.PowerSpectrum()

	k = np.logspace(-3,2,10000)
	dk =  np.gradient(k)
	#pkk = pk(k, 0.475)
	pars = {'method': 'halofit','tracer1': {'b': lambda z: b1}, 'tracer2': {'b': lambda z: b2}}
	p_auto, _, _, _ = pk(pars, k, 0.5*(zmin + zmax))

	#r = np.logspace(-1,2,100)
	xi = 1./(2*np.pi**2.) * np.sum(dk * k**2 * p_auto * np.sin(k * r[:,np.newaxis])/(k * r[:,np.newaxis]),axis=1)
	return xi
	
# Adjustable Options
def wp(b1, b2, zmin, zmax, pairwise=True):
	#color = 'blue'
	#spec = 'unwise_DR12_cmass'
	#zmin = 0.45
	#zmax = 0.50
	#dz = 0.05
	#swise = 0.453

	# Cosmology
	p18_cosmo = FlatLambdaCDM(H0=67.74,Om0=0.3075,Ob0=0.0486,Tcmb0=2.7255,Neff=3.046, m_nu = [0, 0, 0.06] * u.eV)

	# Load the power spectrum
	sys.path.insert(0, '/project/6033532/jptlawre/')
	pk = L.PowerSpectrum()

	# Load the redshift distributions
	#os.chdir('../')
	#dndz = xmatch_dndz(color)


	# lmax = 50000
	ell = np.linspace(0,5e4,int(5e4+1)) # 0.5% convergence compared to 1 point per ell


	#data = np.loadtxt('data/combined_data.txt')
	#R = data[:,0][7:]
	#distance = 0.5 * (Planck15.comoving_distance(zmin) + Planck15.comoving_distance(zmax))


	# Set dndz for the spec sample
	# z_spec = np.linspace(0,4,1000)
	#dndz_spec_all = tracer_dndz(spec,zmin,zmax)
	
	k = np.logspace(-3,2,100000)
	dk =  np.gradient(k)


	pars = {'method': 'halofit','tracer1': {'b': lambda z: b1}, 'tracer2': {'b': lambda z: b2}}
	p_auto, _, _, _ = pk(pars, k, 0.5*(zmin + zmax))
	Om0 = 0.3075
	Omz = (Om0* (1 + 0.5 * (zmin+zmax))**3.)/((Om0) * (1 + 0.5 * (zmin+zmax))**3. + (1 -Om0))
	f = Omz**0.55
	mu = np.linspace(-1,1,1000)
	rsd_factor = (1 + (f/b1) * mu**2)*(1 + (f/b2) * mu**2)
	dmu = mu[1]-mu[0]
	ells = [0, 2, 4]
	integrated_rsd_factors = []
	r = np.logspace(-1,3,500)
	xi = np.zeros((len(r),len(mu)))
	beta = f/b1
	for ell in ells:
		#integrated_rsd_factor = (2 * ell + 1)/2 * np.sum(rsd_factor * legendre(ell)(mu) * dmu)
		if ell == 0:
			alpha_2l = 1 + (2./3.) * beta + (1./5.) * beta**2.
		elif ell == 2:
			alpha_2l = (4./3.) * beta + (4./7.) * beta**2.
		elif ell == 4:
			alpha_2l = (8./35.) * beta**2.
		xi_ell = np.real((1j)**ell * 1./(2*np.pi**2.) * np.sum(dk * k**2 * p_auto * spherical_jn(ell, k * r[:,np.newaxis]),axis=1))
		xi += xi_ell[:,np.newaxis] * legendre(ell)(mu) * alpha_2l
	

		
	#xi = 1./(2*np.pi**2.) * np.sum(dk * k**2 * p_auto * np.sin(k * r[:,np.newaxis])/(k * r[:,np.newaxis]),axis=1)
	rp = np.linspace(0, 100, 795)
	pimax = 100.
	pi = np.linspace(-pimax, pimax, 800)
	dpi = pi[1]-pi[0]
	Rp, Pi = np.meshgrid(rp, pi)
	R = (Rp**2 + Pi**2)**0.5
	xi_spl = RectBivariateSpline(r, mu, xi)
	Mu = Pi/R
	Mu[0] = 0
	xi_spl_redshift_space = xi_spl(R, Mu, grid=False)
	dpi = pi[1] - pi[0]
	print(np.shape(R))
	'''if pairwise:
		zc = 0.5 * (zmin + zmax)
		Hubble = 100 * np.sqrt(Om0 * (1 + zc)**3 + (1-Om0))
		velocity = np.linspace(-2000,2000,1000)
		sigma_v = 400.
		pi_vpec = pi[:,np.newaxis] - velocity * (1 + zc)/Hubble
		print('pi',pi)
		print('velocity',velocity * (1 + zc)/Hubble)
		fv = 1./(sigma_v * 2**0.5) * np.exp(-np.sqrt(2.)*np.abs(velocity)/sigma_v)
		xi_spl_redshift_space_int = np.zeros((len(pi), len(rp)))
		for i in range(len(velocity)):
			Rp, Pi = np.meshgrid(rp, pi_vpec[:,i])
			R = (Rp**2 + Pi**2)**0.5
			Mu = Pi/R
			xi_spl_redshift_space = xi_spl(R, Mu, grid=False)
			dv = velocity[1]-velocity[0]
			print(np.shape(xi_spl_redshift_space_int))
			print(np.shape(xi_spl_redshift_space))
			xi_spl_redshift_space_int += xi_spl_redshift_space * fv[i] * dv'''
	#return rp, np.sum(xi_spl(R), axis=0) * 2 * dpi
	return rp, np.sum(xi_spl_redshift_space, axis=0) * dpi
	'''r = np.logspace(-3,3,500)
	np.savetxt('pk_from_ak.txt',np.array([k,p_auto]).T)
	xi_real = 1./(2*np.pi**2.) * np.sum(dk * k**2 * p_auto * np.sin(k * r[:,np.newaxis])/(k * r[:,np.newaxis]),axis=1)
	np.savetxt('r.txt',r)
	np.savetxt('xi_real.txt',xi_real)
	xi_spl = Spline(r, xi_real)
	rp = np.linspace(0, 100, 795)
	pimax = 100.
	pi = np.linspace(-pimax, pimax, 800)
	dpi = pi[1]-pi[0]
	Rp, Pi = np.meshgrid(rp, pi)
	R = (Rp**2 + Pi**2)**0.5

	return rp, np.sum(xi_spl(R), axis=0)  * dpi
	#return r, xi'''

bias_list = np.linspace(2,2.15,20)
chi2s = np.zeros_like(bias_list)

data = np.loadtxt('../data/rodriguez_torres_fig10.csv',delimiter=',')

#b = np.logspace(np.log10(thmin.value),np.log10(thmax.value),nbins+1,endpoint=True)
bin_zero_min = np.log10(data[0,0])-0.5 * np.mean(np.gradient(np.log10(data[:,0])))
bin_last_max = np.log10(data[-1,0])+0.5 * np.mean(np.gradient(np.log10(data[:,0])))
bins_rp = np.logspace(bin_zero_min, bin_last_max, len(data)+1)
print(bins_rp)

#integral_constraint = 0.04
integral_constraint = 0.0

'''for j,bias in enumerate(bias_list):
	#bias = 2.071
	rp, wpout = wp(bias, bias, 0.43, 0.70)
	wpspline = Spline(rp, wpout)

	data_file = np.loadtxt('../data/cmass_data_0.43_0.7_rodriguez_torres_bins_to_use.txt')
	#data_file = np.loadtxt('../data/boss_0.43_0.7_result.txt').T[2:,:]
	#data_file = np.loadtxt('../data/Rodriguez-Torres-HOD-fit.csv',delimiter=',')[2:,:]
	covariance_file = np.loadtxt('../data/cmass_data_0.43_0.7_rodriguez_torres_covariance_bins_to_use.txt')


	rp_data = data_file[2:,0]
	wp_data = data_file[2:,1]#/rp_data
	wp_theory = wpspline(rp_data)
	#chi2 = np.sum((wp_data-wp_theory)**2./np.diag(covariance_file)[2:])
	#chi2s[i] = chi2
	#print(chi2)
	wp_theory_binned = np.zeros(len(bins_rp)-1)
	for i in range(len(bins_rp)-1):
		rp_big = np.linspace(bins_rp[i],bins_rp[i+1],1000)
		wp_theory_binned[i] = (np.sum(2 * rp_big * (1 + wpspline(rp_big))*np.gradient(rp_big))/((bins_rp[i+1]**2.-bins_rp[i]**2.))-1.)

	chi2s[j] = np.sum(((wp_data-(wp_theory_binned[4:]-integral_constraint))**2./np.diag(covariance_file)[2:])[6:])
	print(chi2s[j])'''
	
#bias = bias_list[np.argmin(chi2s)]

bias = 1.95

rp, wpout = wp(bias, bias, 0.43, 0.70)
wpspline = Spline(rp, wpout)

data_file = np.loadtxt('../data/cmass_data_0.43_0.7_rodriguez_torres_bins_to_use.txt')
#data_file = np.loadtxt('../data/boss_0.43_0.7_result.txt').T[2:,:]
#data_file = np.loadtxt('../data/Rodriguez-Torres-HOD-fit.csv',delimiter=',')[2:,:]
covariance_file = np.loadtxt('../data/cmass_data_0.43_0.7_rodriguez_torres_covariance_bins_to_use.txt')


rp_data = data_file[2:,0]
wp_data = data_file[2:,1]#/rp_data
wp_theory = wpspline(rp_data)
#chi2 = np.sum((wp_data-wp_theory)**2./np.diag(covariance_file)[2:])
#chi2s[i] = chi2
#print(chi2)
wp_theory_binned = np.zeros(len(bins_rp)-1)
for i in range(len(bins_rp)-1):
	rp_big = np.linspace(bins_rp[i],bins_rp[i+1],1000)
	wp_theory_binned[i] = (np.sum(2 * rp_big * (1 + wpspline(rp_big))*np.gradient(rp_big))/((bins_rp[i+1]**2.-bins_rp[i]**2.))-1.)




#np.savetxt('wp_halofit_real_space.txt',np.array([rp,wpout]).T)
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt

plt.ion()
plt.show()
plt.figure()
plt.errorbar(rp_data,rp_data*wp_data,(np.diag(covariance_file)[2:])**0.5*rp_data)
plt.plot(rp_data,rp_data*wp_theory,color='r',label='Theory at center datapoint')



wp_theory_binned = np.zeros(len(bins_rp)-1)
for i in range(len(bins_rp)-1):
	rp_big = np.linspace(bins_rp[i],bins_rp[i+1],1000)
	wp_theory_binned[i] = (np.sum(2 * rp_big * (1 + wpspline(rp_big))*np.gradient(rp_big))/((bins_rp[i+1]**2.-bins_rp[i]**2.))-1.)
	
plt.plot(rp_data,rp_data*wp_theory_binned[4:],color='r',label='Binned theory')
plt.xscale('log')
plt.savefig('wp_halofit_test.png')
print(wp_theory_binned)
print(wp_theory)
print(rp_data)
