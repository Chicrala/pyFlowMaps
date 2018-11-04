#!/usr/bin/python
# -*- coding: utf8 -*-



import numpy as np
from flowmaker import flowmaker
from math_Tools import *
import sys

def flow(cube,fwhm_arcsec, pix, t_step, lag=1, reb=1.0):
	"""
	Spanish: Programa para generar las mapas de flujos vectoriales vx, vy.
	English: Script to generate the vector flow maps vx, vy.
	Inputs:
		fwhm: Window size for tracking(arcsec)
		pix: Size of pixel (Instument information)
		t_step: temporal sampling interval in the time series (seconds)
	Optional Inputs:
		reb: the rebinning factor if it is wished.
		lag: he lag between the images to be compared (number of images)
	
	Output: The function returns the velocity maps for vx, vy, and vz, as 
	        well as the calculate from magnitude = sqrt(vx**2+vy**2), in km/s. For 
	        solar images cases, vz = h_m * div(vx,vz), with h_m the mass-flux
	        height scale. (See November et al. 1987; November et al. 1989).
	"""
	
	#************************************************************
	fwhm=fwhm_arcsec/pix

	kmperasec=725#input('Value of kilometers per arcsec:')
	h_m=150#input('mass-flux scale-heigth (November 1989, ApJ,344,494):')

	v_limit=2*reb+reb #cota maxima velocidad en pixeles.
	delta_t=t_step*lag # time-lag in seconds
	factor=pix*kmperasec/delta_t

	#************************************************************
	
	num=cube.shape[0]

	print('Number of images', num)

	print('Calculando flujos horizontales ...')

	vx,vy=flowmaker(cube,lag,fwhm,reb)
	vx=vx.clip(min=-v_limit,max=v_limit) ; vy=vy.clip(min=-v_limit,max=v_limit)

	print('Computes the divergence and vertical velocities ...')
	
	vx_kps=vx*factor 	#vx in km/s
	vy_kps=vy*factor	#vy in km/s
	
	div= divergence(vx_kps,vy_kps)
	vz_kps=h_m*div
	
	mag = np.sqrt(vx_kps*vx_kps + vy_kps*vy_kps)
	
	
	
	return vx_kps, vy_kps, vz_kps, mag
	
	


