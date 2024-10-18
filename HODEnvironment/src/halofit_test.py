import numpy as np
from astropy.cosmology import FlatLambdaCDM, Planck15
from scipy import special
from scipy.interpolate import InterpolatedUnivariateSpline as Spline
import sys

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