#!/usr/bin/python
# -*- coding: utf8 -*-


#Class flowmaker
import numpy as np
import sunpy
from sunpy.image.rescale import resample
from math_Tools import *
from scipy.ndimage import uniform_filter

__all__ = ['flowmaker']
__authors__ = ["Jose Ivan Campos Rozo, Santiago Vargas Dominguez"]
__email__ = "hypnus1803@gmail.com"



def flowmaker(mov,lag,fwhm,reb,keyword='none'):
	"""
	Compute flow maps and returns X and Y components for the proper 
	motion map.
	Inputs:
	-------
			mov: 3-D array with series of image
			lag: time-lag between 'references' and 'life' subseries.
			fwhm: fwhm for smoothing window in pixels.
			reb: rebinning	factor to change scale/range of November's
			method.
	
	Keywords:
	---------
			boxcar: if set, a boxcar window of width "fwhm" is used. Hence,
			        FWHM must be an odd number.
			adif: uses an absolute differences algorithm.
			corr: uses a multiplicative algorithm. Default is the sum of
				  square of the local differences.
			qfit2: uses 9 points fitting procedure.
			crossd: uses cross derivative interpolation formulae
	Example:
	--------
			>>> vx,vy=flowmaker(cube,1,8,1)
			
	"""
	shf=1
	# ~ std1=fwhm/(2*np.sqrt(2*np.log(2)))
	std1=fwhm*0.424661
	std2=int(std1/reb)


	dims = np.ndim(mov)
	n_im = mov.shape[0]
	yy = mov.shape[1]
	yy_r = int(yy/reb)
	xx = mov.shape[2]
	xx_r = int(xx/reb)
	
	if dims != 3:
		raise ValueError('Array must be 3-dimensional! Breaked')
		
	
	n=int(n_im-lag)
	
	n_p=xx_r*yy_r
	
	cc=np.zeros((3,3,yy_r,xx_r))
	
	for k in range(n):
		a=resample(mov[k,:,:],(yy_r, xx_r),method='neighbor',minusone=False)
		b=resample(mov[k+lag,:,:],(yy_r, xx_r),method='neighbor',minusone=False)
		a=a-np.sum(a)/n_p
		b=b-np.sum(b)/n_p
		
		for i in range(-1,2):
			for j in range(-1,2):
				
				if keyword=='adif':
					cc[j+1,i+1,:,:]=cc[j+1,i+1,:,:]+abs(np.roll(a,(i*shf,j*shf),axis=(1,0))-np.roll(b,(-i*shf,-j*shf),axis=(1,0)))
				
				if keyword=='corr':
					cc[j+1,i+1,:,:]=cc[j+1,i+1,:,:] + np.roll(a,(i*shf,j*shf),axis=(1,0)) - np.roll(b,(-i*shf,-j*shf),axis=(1,0))
				
				else:
					dumb = np.roll(a,(i*shf,j*shf),axis=(1,0)) - np.roll(b,(-i*shf,-j*shf),axis=(1,0))
					
					cc[j+1,i+1,:,:]=cc[j+1,i+1,:,:]+dumb*dumb
					dumb = 0
		a = 0 
		b = 0
	
	cc[:,:,:,0]=cc[:,:,:,1]
	cc[:,:,0,:]=cc[:,:,1,:]
	cc[:,:,:,xx_r-1]=cc[:,:,:,xx_r-2]
	cc[:,:,yy_r-1,:]=cc[:,:,yy_r-2,:]
	
	for i in range(3):
		for j in range(3):
			if keyword=='boxcar':
				cc[j,i,:,:]=uniform_filter(cc[j,i,:,:],int(fwhm/reb))
			cc[j,i,:,:]=sconvol1d(cc[j,i,:,:],std=std2)
	
	if keyword=='qfit2':
		vx,vy=qfit2(cc)
	elif keyword=='crossD':
		vx,vy=crossD(cc)
	else:
		vx,vy=fivepoint(cc)

	vx=2.*shf*vx
	vy=2.*shf*vy
	vx=resample(vx,(yy,xx),center=True,method='neighbor',minusone=False)*reb
	vy=resample(vy,(yy,xx),center=True,method='neighbor',minusone=False)*reb
	
	return vx,vy




		
