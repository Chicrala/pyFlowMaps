#!/usr/bin/python
# -*- coding: utf8 -*-
import numpy as np
from scipy.ndimage import correlate1d
import numbers



__all__ = ['sconvol1d','fivepoint','qfit2','crossD','divergence']
__authors__ = ["Jose Ivan Campos Rozo, Santiago Vargas Dominguez"]
__email__ = ["hypnus1803@gmail.com","svargasd@unal.edu.co"]



def sconvol1d(arreglo,kernel='none',scale_factor=1.0,std='none',fwhm='none',**kwargs):
	"""
	This program will smooth a 2D array, including the edges,
	with one-dimensional kernels. Problems of this kind arise when,
	e.g. an array is to be convolved with a 2D symmetric
	gaussian, which is separable into two one-dimensional
	convolutions.
	Similar to SCONVOL from IDL: R.Molowny-Horas and Z.Yi, May 1994.
	"""
	
	if kernel == 'none':
		if std == 'none' and fwhm == 'none':
			raise ValueError('There is not a way to convolve. Set kernel, std, or fwhm')
		
		if (std != 'none') and (isinstance(std,numbers.Number)) and (fwhm == 'none'):
			width = np.floor(std*9.)
		if (fwhm != 'none') and (isinstance(fwhm,numbers.Number)):
			std = fwhm/(2*np.sqrt(2*np.log(2)))
			width = np.floor(std*9.)
		if width%2 == 0: width += 1.
		kernel = np.arange(width) -np.floor(width/2.)
		kernel = np.exp(-kernel*kernel/(2*std**2))
		kernel = kernel/(std*np.sqrt(2*np.pi))
	
	
	elif isinstance(kernel,numbers.Number):
		kernel = np.array([kernel])
	elif len(kernel)%2 == 0:
		raise ValueError('Length of kernel must be odd')
	

	
	return np.rot90(correlate1d(np.rot90(correlate1d(arreglo,kernel,**kwargs),3),kernel,**kwargs),1)



def fivepoint(cc):
	""" Purpose: 
		Measure the position of minimum or maximum in a 3x3 matrix.
		Written by Roberto Luis Molowny Horas, Institute of Theoretical
		Astrophysics, University of Oslo. August 1991.
		This version is based on the IDL version.
		Adapted in Python by J. I. Campos-Rozo
	"""
	
	dim = np.ndim(cc)
	
	if dim < 2 or dim>4:
		raise ValueError('Wrong input array dimensions')
	if cc.shape[0] != 3 or cc.shape[1] != 3:
		raise ValueError('Array-shape must be CC[3,3,:,:], or CC[3,3,:], or CC[3,3]')
	
	if dim == 4:
		y = 2*cc[1,1,:,:]
		x = (cc[1,0,:,:]-cc[1,2,:,:])/(cc[1,2,:,:]+cc[1,0,:,:]-y)*0.5
		y=(cc[0,1,:,:]-cc[2,1,:,:])/(cc[2,1,:,:]+cc[0,1,:,:]-y)*0.5
	
	
	elif dim == 3:
		y=2.*cc[1,1,:]
		x=(cc[1,0,:]-cc[1,2,:])/(cc[1,2,:]+cc[1,0,:]-y)*0.5
		y=(cc[0,1,:]-cc[2,1,:])/(cc[2,1,:]+cc[0,1,:]-y)*0.5
	
	elif dim == 2:
		y=2.*cc[1,1]
		x=(cc[1,0]-cc[1,2])/(cc[1,2]+cc[1,0]-y)*0.5
		y=(cc[0,1]-cc[2,1])/(cc[2,1]+cc[0,1]-y)*0.5
	
	return x,y



def qfit2(cc):
	""" Purpose:
		Measure the position of extrem value in a 3x3 matrix
		Taken from the IDL version written by R. Molowny Horas.
		Adapated for Python by J. I. Campos-Rozo
	"""

	
	dim = np.ndim(cc)


	if (dim < 2) and (dim > 4):
		raise ValueError('Wrong input array dimasions')
	if cc.shape[0] != 3 or cc.shape[1] != 3:
		raise ValueError('Array-shape must be CC[3,3,:,:], or CC[3,3,:], or CC[3,3]')

	if dim == 4 :
		a1=cc[0,0,:,:]+cc[0,2,:,:]+cc[2,0,:,:]+cc[2,2,:,:]
		a2 = a1+cc[0,1,:,:]+cc[2,1,:,:]
		a1 = a1+cc[1,0,:,:]+cc[1,2,:,:]
		a3 = cc[0,0,:,:]-cc[0,2,:,:]-cc[2,0,:,:]+cc[2,2,:,:]
		a4 = -cc[0,0,:,:]+cc[2,2,:,:]
		a5 = a4-cc[0,1,:,:]-cc[0,2,:,:]+cc[2,0,:,:]+cc[2,1,:,:]
		a4 = a4+cc[0,2,:,:]-cc[1,0,:,:]+cc[1,2,:,:]-cc[2,0,:,:]
		a1 = .5*a1-cc[0,1,:,:]-cc[1,1:,:]-cc[2,1,:,:]
		a2 = .5*a2-cc[1,0,:,:]-cc[1,1,:,:]-cc[1,2,:,:]
	elif dim == 3:
		a1 = cc[0,0,:]+cc[0,2,:]+cc[2,0,:]+cc[2,2,:]
		a2 = a1+cc[0,1,:]+cc[2,1,:]
		a1 = a1+cc[1,0,:]+cc[1,2,:]
		a3 = cc[0,0,:]-cc[0,2,:]-cc[2,0,:]+cc[2,2,:]
		a4 = -cc[0,0,:]+cc[2,2,:]
		a5 = a4-cc[0,1,:]-cc[0,2,:]+cc[2,0,:]+cc[2,1,:]
		a4 = a4+cc[0,2,:]-cc[1,0,:]+cc[1,2,:]-cc[2,0,:]
		a1 = .5*a1-cc[0,1,:]-cc[1,1,:]-cc[2,1,:]
		a2 = .5*a2-cc[1,0,:]-cc[1,1,:]-cc[1,2,:]
	elif dim == 2:
		a1=cc[0,0]+cc[0,2]+cc[2,0]+cc[2,2]
		a2=a1+cc[0,1]+cc[2,1]
		a1=a1+cc[1,0]+cc[1,2]
		a3=cc[0,0]-cc[0,2]-cc[2,0]+cc[2,2]
		a4=-cc[0,0]+cc[2,2]
		a5=a4-cc[0,1]-cc[0,2]+cc[2,0]+cc[2,1]
		a4=a4+cc[0,2]-cc[1,0]+cc[1,2]-cc[2,0]
		a1=.5*a1-cc[0,1]-cc[1,1]-cc[2,1]
		a2=.5*a2-cc[1,0]-cc[1,1]-cc[1,2]

	dim_ = ((64./9)*a1*a2-a3**2)*1.5

	cx = (a3*a5-((8./3)*a2*a4))/dim_
	cy = (a3*a4-8./3*a1*a5)/dim_
	return cx, cy

def crossD(cc):
	""" Purppose:
				Measure the position of a minimum or maximum in a 3x3 array. This function
				is an extension of the five-point function, but, taking into account the
				four points in the corners.
				Written by Roberto Luis Molowny Horas, August 1991.
				Adapted for Python by J. I. Campos-Rozo
	Example:
	--------
	>>> c=dist(3,3)
	
	>>> crossD(c)
	(0.369398, 0.369398)
	
	"""
	dim = np.ndim(cc)


	if (dim < 2) and (dim > 4):
		raise ValueError('Wrong input array dimasions')
	if cc.shape[0] != 3 or cc.shape[1] != 3:
		raise ValueError('Array-shape must be CC[3,3,:,:], or CC[3,3,:], or CC[3,3]')

	if dim == 2:
		c4=cc[1,2]+cc[1,0]-cc[1,1]*2.
		c2=cc[1,2]-cc[1,0]
		c5=cc[2,1]+cc[0,1]-cc[1,1]*2.
		c3=cc[2,1]-cc[0,1]
		c6=(cc[2,2]-cc[2,0]-cc[0,2]+cc[0,0])/4.
	elif dim==3:
		c4=cc[1,2,:]+cc[1,0,:]-cc[1,1,:]*2.
		c2=cc[1,2,:]-cc[1,0,:]
		c5=cc[2,1,:]+cc[0,1,:]-cc[1,1,:]*2
		c3=cc[2,1:]-cc[0,1,:]
		c6=(cc[2,2,:]-cc[2,0,:]-cc[0,2,:]+cc[0,0,:])/4.
	elif dim ==4:
		c4=cc[1,2,:,:]+cc[1,0,:,:]-cc[1,1,:,:]*2.
		c2=cc[1,2,:,:]-cc[1,0,:,:]
		c5=cc[2,1,:,:]+cc[0,1,:,:]-cc[1,1,:,:]*2
		c3=cc[2,1:,:]-cc[0,1,:,:]
		c6=(cc[2,2,:,:]-cc[2,0,:,:]-cc[0,2,:,:]+cc[0,0,:,:])/4.
	determ=0.5/(c4*c5 - c6*c6)
	x=determ*(c6*c3 - c5*c2)
	y=determ*(c6*c2 - c4*c3)
	return x, y



def divergence(vx,vy):
	
	"""
		PURPOSE:
			Make divergence of between two 2-D velocity map.
			v_z = div(vx,vy)
	
	
	"""
	X, Y = np.meshgrid(np.arange(0,vx.shape[1]),np.arange(0,vx.shape[0]))
	du_x,du_y = np.gradient(vx,Y[:,1],X[1,:])
	dv_x,dv_y = np.gradient(vy,Y[:,1],X[1,:])
	div = dv_x+du_y
	
	return div

