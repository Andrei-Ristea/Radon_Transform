##################################################################################################################################

# v 9.0
'Based on the method in Stark et al. 2018 (initially applied to galaxies in the MAnGA survey)'
# This script implements the Radon Transform line integral method in Python and makes it applicable to any type of IFS data

# Functions:
#_____________________________________________________________________________________
# --- generate_ifu_map ---
# Generates a model velocity map to be used for testing the Radon statistic functions
#_____________________________________________________________________________________
#
#_____________________________________________________________________________________
# --- radon ---
# Runs on IFS velocity maps
# Computes the Radon/Abs. Radon/Abs. Bounded Radon (for given radon apperture) statistics (and rescaled to 0-1 versions) corresponding to the givel vel. map
# Extracts Radon profiles thata_hat(rho) - see Stark et al. (2018)
#_____________________________________________________________________________________


# Update to v 2.0 (from 1.0):
#------------------------------
# Exclude areas in the R_AB
#------------------------------

# Update to v 3.0 (from 2.0):
#------------------------------
# Add a function that fits a slice through the R_AB (abs. bounded Radon) map at fixed rho with a von Mises function
#------------------------------

#                   v 4.0 --- v 5.0

# Update to v 6.0 (from 5.0):
#------------------------------
# Optimize the part of the 'radon' function which computes radon maps - in the previous version,
# this was inefficinet and took ~30s to run for one galaxy.
#------------------------------

# Update to v 7.0 (from 6.0):
#------------------------------
# Includes error calculations for R_AB and theta_hat in Radon profiles
#------------------------------

##################################################################################################################################

from __future__ import division
import numpy
import matplotlib.pyplot as plt
import random 
import scipy
import scipy.optimize as optimization
from scipy import special
from scipy import stats
import matplotlib.patches as mpatches

#import lmfit
#from lmfit import Model
#from lmfit import minimize, Parameters, Parameter, report_fit
import matplotlib.colors as colors
import matplotlib.cm
from astropy.io import fits
from numpy import *

from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.gridspec as gridspec
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
#import skimage.util
#from skimage.transform import rotate
from scipy.ndimage import rotate
import scipy.ndimage as sn

from matplotlib.colors import LogNorm
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple
import amstools_2
import pafit_fit_kinematic_pa
import os
import warnings
from astropy.table import Table , hstack

import loess_1d

from collections import deque,Counter
from bisect import insort, bisect_left
from itertools import islice

#from chainconsumer import ChainConsumer
from scipy.stats import norm
#import emcee

from scipy.signal import argrelextrema
import time
start_time = time.time()

#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________

#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
# Various Bayesian function needed for implementing the Radon transform method 
#________________________________________________________________________________________________________________________________________________________________

def log_prior(xs):
    phi, c = xs
    if numpy.abs(phi) > numpy.pi / 2:
        return - numpy.inf
    return 0

def log_likelihood(xs, data):
    phi, c = xs
    xobs, yobs, eobs = data
    model = numpy.tan(phi) * xobs + c
    diff = model - yobs
    return norm.logpdf(diff / eobs).sum()

def log_posterior(xs, data):
    prior = log_prior(xs)
    if not numpy.isfinite(prior):
        return prior
    return prior + log_likelihood(xs, data)

def fit_linear(x,y,err, ndim = 1 ,nwalkers = 50):
    p0 = numpy.random.uniform(low=-1.5, high=1.5, size=(nwalkers, ndim))  # Start points
    #sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=[(x, y, err)])
    

#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________

def coord_plot_PAkin(angle_max, angle_lower, angle_upper, extent_xmin, extent_xmax):
    #________________________________________________________________________________________________________________________________
    # Given a position angle and [x0,x1] coordinates, the function computes the [y0,y1] coordinates for plotting the axis
    # NOTE: angle_max must be between [0,180] and measured from EAST = 0 Counterclockwise (i.e. before any corrections)
    #       Must correct the value of angle_max otherwise
    #________________________________________________________________________________________________________________________________
    #    ARGUMENTS :
    # --------------------------------------------------------------------------------------------------------------------------------
    #    angle_max     :  Position angle of the axis that needs to be plotted
    #   angle_lower    :  Lower error angle in angle_max - for plotting the error interval around the axis
    #   angle_upper    :  Upper error angle in angle_max - for plotting the error interval around the axis
    #   extent_xmin    :  Lower limit on the x-axis of the interval over which the axis needs to be represented
    #   extent_xmax    :  Upper limit on the x-axis of the interval over which the axis needs to be represented
    # _______________________________________________________________________________________________________________________________
    #     RETURNS  :
    #-------------------------------------------------------------------------------------------------------------------------------
    #  ymin     : Lower y coordinate for plotting the axis determined by angle_max
    #  ymax     : Upper y coordinate for plotting the axis determined by angle_max
    #  ya_lower : = numpy.array([ymin_ls,ymax_ls]) -- 1d array of the lower/upper coordinates of the line showing the lower error in PA_Kin
    #  ya_upper : = numpy.array([ymin_us,ymax_us]) -- 1d array of the lower/upper coordinates of the line showing the upper error in PA_Kin
    #________________________________________________________________________________________________________________________________
    if angle_max == -1:
        return 0,0,0,0,0,0

    'For plotting the axis determined by angle_max:'
    angle_max_RAD = (angle_max*numpy.pi)/180

    if angle_max > 90:
        ymin = -extent_xmin*numpy.tan(numpy.pi-angle_max_RAD)
        ymax = -extent_xmax*numpy.tan(numpy.pi-angle_max_RAD)
    else:
        ymin = extent_xmin*numpy.tan(angle_max_RAD)
        ymax = extent_xmax*numpy.tan(angle_max_RAD)

    'For plotting the error interval around the axis, determined by angle_lower & angle_upper:'
    angle_lowerRAD = (angle_lower*numpy.pi)/180
    angle_upperRAD = (angle_upper*numpy.pi)/180

    if angle_lower > 90:
        ymin_ls= - extent_xmin*numpy.tan(numpy.pi-angle_lowerRAD)
        ymax_ls= - extent_xmax*numpy.tan(numpy.pi-angle_lowerRAD)
    else:
        ymin_ls = extent_xmin*numpy.tan(angle_lowerRAD)
        ymax_ls = extent_xmax*numpy.tan(angle_lowerRAD)
    #_____________________________________________________
    if angle_upper>90:
        ymin_us = -extent_xmin*numpy.tan(numpy.pi-angle_upperRAD)
        ymax_us = -extent_xmax*numpy.tan(numpy.pi-angle_upperRAD)
    else:
        ymin_us = extent_xmin*numpy.tan(angle_upperRAD)
        ymax_us = extent_xmax*numpy.tan(angle_upperRAD)
    ya_lower=numpy.array([ymin_ls,ymax_ls])
    ya_upper=numpy.array([ymin_us,ymax_us]) 

    return ymin, ymax, ymin_ls, ymax_ls, ymin_us, ymax_us


def generate_ifu_velmap(x,y,cx,cy,inclin,v_asympt,re,PA,dPA,pixelres):
    #__________________________________________________________
    #    x,y    : Dimensions of the map to be generated | in pixels
    #   cx,cy   : Offset of the centre of the vel. map in the x and y-directions | in pixels (if cx=cy=0, map is centred on the [0,0] point of the image)
    #   inclin  : Inclination angle (~b/a) | in degrees
    #  v_asympt : Asymptotic rotational velocity | in km/s
    #    re     : Effective radius | in arcsec
    #    PA     : Position angle of the main kinematic axis | in deg
    #  pixelres : Resolution of one pixel | in arcsec
    #    dPA    : Change in the Kin. pos angle (PA) to generate warped velocity field | in deg/Re
    #___________________________________________________________
    #  h = re/2 : Scale length of rotation curve
    #-----------------------------------------------------------
    # Form of the velocity map - one of the two below:
    # 1) Normal rotating disk : v[x,y] = v0*tanh(r/h)*sin(i)cos(phi-PA_Kin)
    # 2) ---
    #__________________________________________________________
    vel_map = numpy.zeros([x,y])
    h = re/2
    xc      = int(x/2) + cx
    yc      = int(y/2) + cy

    r_max = 10 # arcsec
    
    inclin = numpy.pi*(inclin/180) # Conversion to radians
    
    # 

    x_arr   = numpy.linspace(0,x,num=x,endpoint=True,dtype=int) #- cx
    y_arr   = numpy.linspace(0,y,num=y,endpoint=True,dtype=int) #- cy

    x_coord = x_arr*pixelres - (x*pixelres/2) + pixelres/2
    y_coord = y_arr*pixelres - (y*pixelres/2) + pixelres/2

    r_arr = numpy.zeros([x,y])
    for i in range(x):
        for j in range(y):
            r_arr[i,j] = numpy.sqrt((x_coord[i]-cx*pixelres)**2 + (y_coord[j]-cy*pixelres)**2)
           
    
    phi_arr = numpy.zeros([x,y])
    for i in range(x):
        for j in range(y):
            if i<=(int(x/2)+cx-1) and j>(int(y/2)+cy-1):
                phi_arr[i,j] = numpy.arctan(abs(y_coord[j]-cy*pixelres)/abs(x_coord[i]-cx*pixelres)) + numpy.pi
            if i<=(int(x/2)+cx-1) and j<=(int(y/2)+cy-1):
                phi_arr[i,j] = numpy.arctan(abs(x_coord[i]-cx*pixelres)/abs(y_coord[j]-cy*pixelres)) + numpy.pi/2
            if i>(int(x/2)+cx-1) and j<=(int(y/2)+cy-1):
                phi_arr[i,j] = numpy.arctan(abs(y_coord[j]-cy*pixelres)/abs(x_coord[i]-cx*pixelres))
            if i>(int(x/2)+cx-1) and j>(int(y/2)+cy-1):
                phi_arr[i,j] = numpy.arctan(abs(x_coord[i]-cx*pixelres)/abs(y_coord[j]-cy*pixelres)) + 3*numpy.pi/2
            
    phi_arr = (180/numpy.pi)*phi_arr

    for i in range(x):
        for j in range(y):
            arg = phi_arr[i,j] - PA - dPA*(r_arr[i,j]/re) # in degrees
            arg = numpy.pi*(arg/180)
            if r_arr[i,j]<r_max:
                vel_map[i,j] = v_asympt*numpy.tanh(r_arr[i,j]/h)*numpy.sin(inclin)*numpy.cos(arg)
            else:
                vel_map[i,j] = numpy.nan
                
    phi_arr_180 = numpy.copy(phi_arr)
    for i in range(x):
        for j in range(y):
            if phi_arr_180[i,j]>=180:
                phi_arr_180[i,j] = phi_arr_180[i,j] - 180


    return vel_map,     phi_arr,      phi_arr_180,         r_arr


vel_map,   phi_arr,   phi_arr_180    ,r_arr = generate_ifu_velmap(50,50,0,0,20,200,3,135,0,0.5)


#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________


def von_mises(x,k,a,mu,b,sl):
    # To implement the mathematical Von Mises function
    #________________________________________________
    I0 = scipy.special.i0(k)
    funct  = (   a*numpy.exp( k*numpy.cos(x-mu) ) / (2*numpy.pi*I0)   )   +   b    +  sl*x

    return funct


def optimise_vm(x,y,guess,error,bds): 
    # To fit a von Mises function to data
    #________________________________________________
    array = optimization.curve_fit(von_mises,x,y,guess,sigma=error,absolute_sigma=True,bounds=bds)
    return array



def asymmetry(theta,rho,N00):
    # To calculate the asymmetry of a 2D velocity field along a specified angle (theta), and at a specific radius (rho)
    #________________________________________________
    if len(theta)!=0 and len(rho)!=0:
        theta_flip = numpy.flip(theta)
        rho_flipped = numpy.flip(-1*(rho))

        Asym = 0
        Nij  = 0#len(theta)
        #wij  = N00/Nij

        rho_common = []
        for i in range(len(rho)):
            for j in range(len(rho_flipped)):
                if numpy.around(rho[i],decimals=3)  ==  numpy.around(rho_flipped[j],decimals=3):
                    Asym = Asym + abs(theta[i] - theta_flip[j])
                    rho_common.append(rho[i])
                    Nij = Nij + 1
        
        rho_common  = numpy.array(rho_common)
        try:
            rho_com_min = numpy.nanmin(rho_common)
        except:
            rho_com_min = -1
        try:
            rho_com_max = numpy.nanmax(rho_common)
        except:
            rho_com_max = -1
        'Test if there is any overlap at all between theta & theta_flip (if there is none, Nij = 0)'
        # NOTE: Only consider radon profile for asymmetry calculation if the overlab between original & flipped arrays is of AT LEAST 4 data points
        if Nij >= 4:                    
            wij  = N00/Nij
            Asym = (Asym*wij)/(2*Nij)
        else:
            wij = 0
            Asym = 99

        return Asym, Nij, wij, rho_com_min, rho_com_max
    else:
        return -1, -1, -1, -1, -1

def shift_image(velmap, pixelres,shift, apperture = 2,plot='no'):
    # Shifts a velocity map (velmap) with a specified pixel resolution a number of pixels in a certain direction
    #__________________________________________________________________
    #__________________________________________________________________
    x_dim        = len(velmap[:,0])   # = 50 for SAMI
    y_dim        = len(velmap[0,:])   # = 50 for SAMI
    x_dim_arcsec = x_dim*pixelres     # = 25'' for SAMI
    y_dim_arcsec = y_dim*pixelres     # = 25'' for SAMI
    
    ymin_arcsec = -(y_dim*pixelres - pixelres)/2 
    ymax_arcsec = (y_dim*pixelres - pixelres)/2
    xmin_arcsec = -(x_dim*pixelres - pixelres)/2 
    xmax_arcsec = (x_dim*pixelres - pixelres)/2
    #__________________________________________________________________
    #__________________________________________________________________
    velmap_shifted = scipy.ndimage.shift(velmap, shift=shift, order=1, mode = 'constant',cval=numpy.nan)
    #__________________________________________________________________
    #__________________________________________________________________
    if plot == 'yes':
        figr,axsr = plt.subplots(1,2,figsize=(40,40))

        csfont = {'fontname':'Times New Roman'}

        vmr0 = axsr[0].imshow(velmap, origin='lower',cmap='RdYlBu_r',aspect='equal',extent=[-x_dim_arcsec/2,x_dim_arcsec/2,-y_dim_arcsec/2,y_dim_arcsec/2],vmin=-250,vmax=250,zorder=0) # vmin=numpy.nanmin(velmap),vmax=numpy.nanmax(velmap)
        divider0 = make_axes_locatable(axsr[0])
        cax0 = divider0.append_axes('right', size='5%', pad=0.05)
        cbar0=figr.colorbar(vmr0, cax=cax0, orientation='vertical')
        cbar0.ax.tick_params(labelsize=7)
        
        a_circle = plt.Circle((0, 0), apperture,edgecolor='black',fill=False,linestyle='--',linewidth=1.8,zorder=10)
        axsr[0].add_artist(a_circle)
        axsr[0].scatter(0,0,s=90,color='black',marker='o',zorder=20)
        axsr[0].axvline(x=0,color='black',zorder=20,linewidth=1.2)
        axsr[0].axhline(y=0,color='black',zorder=20,linewidth=1.2)
        height, width = pixelres, pixelres
        
        axsr[0].axvline(x=xmax_arcsec,color='navy',zorder=20,linewidth=1.2)
        axsr[0].axhline(y=ymax_arcsec,color='navy',zorder=20,linewidth=1.2)
        axsr[0].axvline(x=xmin_arcsec,color='black',zorder=20,linewidth=1.2)
        axsr[0].axhline(y=ymin_arcsec,color='black',zorder=20,linewidth=1.2)
        axsr[0].set_xlabel('arcsec',fontsize=12,**csfont)
        axsr[0].set_ylabel('arcsec',fontsize=12,**csfont)

        axsr[0].xaxis.set_major_locator(MultipleLocator(5))
        axsr[0].xaxis.set_minor_locator(MultipleLocator(1))
        axsr[0].yaxis.set_major_locator(MultipleLocator(5))
        axsr[0].yaxis.set_minor_locator(MultipleLocator(1))
        axsr[0].tick_params(which='both',direction='in',width=0.5)

        velmap_copy = numpy.copy(vel_map)
        for i in range(x_dim):
            for j in range (y_dim):
                if numpy.isnan(velmap_copy[i,j])==True:
                    velmap_copy[i,j] = 0
        
        #__________________________________________________________________________________________
        vmr1 = axsr[1].imshow(velmap_shifted, origin='lower',cmap='RdYlBu_r',aspect='equal',extent=[-x_dim_arcsec/2,x_dim_arcsec/2,-y_dim_arcsec/2,y_dim_arcsec/2],vmin=numpy.nanmin(velmap_shifted),vmax=numpy.nanmax(velmap_shifted),zorder=0) 
        a_circle = plt.Circle((0 + shift[1]*pixelres, 0 + shift[0]*pixelres), apperture,edgecolor='black',fill=False,linestyle='--',linewidth=1.8,zorder=10)
        axsr[1].add_artist(a_circle)
        axsr[1].scatter(0,0,s=70,color='black',marker='o',zorder=20)
        axsr[1].axvline(x=0,color='black',zorder=20,linewidth=1.2)
        axsr[1].axhline(y=0,color='black',zorder=20,linewidth=1.2)
        #___________
        axsr[1].scatter(0 + shift[1]*pixelres,0 + shift[0]*pixelres,s=70,color='mediumspringgreen',edgecolor='black',marker='o',zorder=20)
        axsr[1].axvline(x=0 + shift[1]*pixelres,color='mediumspringgreen',zorder=20,linewidth=1.2)
        axsr[1].axhline(y=0 + shift[0]*pixelres,color='mediumspringgreen',zorder=20,linewidth=1.2)

        axsr[1].axvline(x=xmax_arcsec,color='navy',zorder=20,linewidth=1.2)
        axsr[1].axhline(y=ymax_arcsec,color='navy',zorder=20,linewidth=1.2)
        axsr[1].axvline(x=xmin_arcsec,color='black',zorder=20,linewidth=1.2)
        axsr[1].axhline(y=ymin_arcsec,color='black',zorder=20,linewidth=1.2)
        axsr[1].set_xlabel('arcsec',fontsize=12,**csfont)
        axsr[1].set_ylabel('arcsec',fontsize=12,**csfont)

    return velmap_shifted 


#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________

def wrap_array(x,y,x_min=numpy.array([-1]),x_max=numpy.array([-1]),plot='No'):
    #_______________________________________________
    # x | y     : coordinates of the function datapoints (y=f(x)) to be wrapped in the x-drection 
    # x_min_max : INDICIES of minima/maxima in the x-array to be wrapped (if wrapping of these is not desired, imput x_min=-1,y_min=-1)
    # NOTE: x and y must be the same length: # 
    # plot  : keyword argument --- if plot = 'yes', a plot is made of the wrapped vs unwrapped function
    #_______________________________________________
    if len(x)!=len(y):
        print('Error --- x and y are not the same size')
    else:
        x_wrap     = numpy.zeros(3*len(x)) # wrap array to the left/right of the imput array --- this is the origin of the 3x multiplication
        y_wrap     = numpy.zeros(3*len(y))
        if all(x_min)!=-1 and len(x_min)!=0:
            x_min_wrap = numpy.zeros(3*len(x_min))
        else:
            x_min_wrap = numpy.zeros(1) -1 
        if all(x_max)!=-1 and len(x_max)!=0:
            x_max_wrap = numpy.zeros(3*len(x_max))
        else:
            x_max_wrap = numpy.zeros(1) -1 
        index     = 0
        index_min = 0
        index_max = 0
        for i in range(len(x_wrap)):
            if i < len(x):
                x_wrap[index] = x[i] - max(x)
                y_wrap[index] = y[i]
                index = index + 1
            elif i>=len(x) and i < 2*len(x):
                x_wrap[index] = x[i-len(x)] 
                y_wrap[index] = y[i-len(x)]
                index = index + 1
            elif i >= 2*len(x):
                x_wrap[index] = x[i-2*len(x)] +  max(x)
                y_wrap[index] = y[i-2*len(x)]
                index = index + 1
        
        if  all(x_min)!=-1 and len(x_min)!=0:
            for i in range(len(x_min_wrap)):
                if i < len(x_min):
                    x_min_wrap[index_min] = x_min[i] #- max(x_min)
                    index_min = index_min + 1
                elif i >= len(x_min) and i < 2*len(x_min):
                    x_min_wrap[index_min] = x_min[i-len(x_min)] + len(x) 
                    index_min = index_min + 1
                elif i >= 2*len(x_min):
                    x_min_wrap[index_min] = x_min[i-2*len(x_min)] + 2*len(x)
                    index_min = index_min + 1
        else:
            x_min_wrap = numpy.zeros(1) -1 

        if  all(x_max)!=-1 and len(x_max)!=0:
            for i in range(len(x_max_wrap)):
                if i < len(x_max):
                    x_max_wrap[index_max] = x_max[i]
                    index_max = index_max + 1
                elif i >= len(x_max) and i < 2*len(x_max):
                    x_max_wrap[index_max] = x_max[i-len(x_max)] + len(x)
                    index_max = index_max + 1
                elif i >= 2*len(x_max):
                    x_max_wrap[index_max] = x_max[i-2*len(x_max)] +  2*len(x)
                    index_max = index_max + 1
        else:
            x_max_wrap = numpy.zeros(1) -1  

    if plot == 'yes':
        fig = plt.figure(figsize=(40,40))
        plt.subplots_adjust(left=0.06, bottom=0.238, right=0.974, top=0.671, wspace=0, hspace=0.102) 
        plt.plot(x,y,color='red',linewidth = 6, linestyle = '-',zorder = 5)
        plt.plot(x_wrap, y_wrap, color='blue',linewidth = 4, linestyle = '-',zorder = -10)
        plt.scatter(x,y,color='red',s=80, marker='s',edgecolor='black',zorder = 10,label='Original x/y')
        plt.scatter(x_wrap,y_wrap,color='blue',s=80, marker='s',edgecolor='black',zorder = 0,label='Wrapped x/y')
        plt.axvline(x=x[0],linestyle='--',color='black',linewidth=7,zorder=-20,alpha=0.4)
        plt.axvline(x=x[len(x)-1],linestyle='--',color='black',linewidth=7,zorder=-20,alpha=0.4)
        plt.xlabel('x',fontsize=14)
        plt.ylabel('y',fontsize=14)
        plt.legend(loc='upper left')
        if len(x_min_wrap) >= 1:
            for i in range(len(x_min_wrap)):
                if x_min_wrap[i] >= max(x_min) and x_min_wrap[i] <= max(x_min) + len(x):
                    plt.axvline(x=x_wrap[int(x_min_wrap[i])],linestyle='-',color='tomato',linewidth=2,zorder=-20)
                else:
                    plt.axvline(x=x_wrap[int(x_min_wrap[i])],linestyle='-',color='cornflowerblue',linewidth=2,zorder=-20)
        if len(x_max_wrap) >= 1:
            for i in range(len(x_max_wrap)):
                if x_max_wrap[i] >= max(x_max) and x_max_wrap[i] <= max(x_max) + len(x):
                    plt.axvline(x=x_wrap[int(x_max_wrap[i])],linestyle='-',color='darkred',linewidth=2,zorder=-20)
                else:
                    plt.axvline(x=x_wrap[int(x_max_wrap[i])],linestyle='-',color='navy',linewidth=2,zorder=-20)

        #plt.show()
    return x_wrap, y_wrap, x_min_wrap, x_max_wrap

#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________

def radon_slice_fit(rd_AB, theta, cutoff , bound_theta=30, theta_guess = 'minima', theta_prev = -1,error = numpy.array([-1]), rho=-1):
    # A function to fit a von Mises function to a certain slice (at fixed theta) in a Radon map
    ' 2) Smooth the <<< R_AB vs theta >>> plot at a given rho'
    try:
        theta_smooth, rd_AB_smooth, w_smooth0 = loess_1d.loess_1d(theta,rd_AB,frac=0.35,npoints=int(0.35*len(rd_AB)),degree=3,rotate=False)
    except:
        print('Loess smoothing failed! Use the original (unsmoothed) arrays!')
        theta_smooth, rd_AB_smooth, w_smooth0 = theta, rd_AB, numpy.zeros(len(theta))-1
    
    ' 3) Compute the positions of the local/global extrema and eliminate peaks/thoughts below the given cutoff'
    cf = cutoff
    grad, ind_minima, minima, peak_width_min, weight_min, norm_weight_min, cutoff_plot, number_minima_after,  minima_cutoff,  ind_minima_cutoff, norm_weight_min_cutoff = extrema(rd_AB_smooth,kw='min',cutoff_extrema=cf)
    grad, ind_maxima, maxima, peak_width_max, weight_max, norm_weight_max, cutoff_plot, number_maxima_after,  maxima_cutoff,  ind_maxima_cutoff, norm_weight_max_cutoff = extrema(rd_AB_smooth,kw='max',cutoff_extrema=cf)

    ' 4) Wrap '
    theta_wrap, rd_AB_wrap, ind_minima_cutoff_wrap, ind_maxima_cutoff_wrap = wrap_array   (theta,  rd_AB,   ind_minima_cutoff,    ind_maxima_cutoff,plot='Yes')


    ' 5) Determine the index of the strongest minima:'
    index_strongest_min = -1
    
    for i in range(len(ind_minima_cutoff)):
        if norm_weight_min_cutoff[i] == max(norm_weight_min_cutoff):
            index_strongest_min = ind_minima_cutoff[i]
    

    ' 6) Choose the interval in <<< theta_wrap >>> over which to fit the von Mises function:'
    # Fit between the two nearest maxima around the STRONGEST MIN, or < +/- bound_theta > deg, whichever is smaller # 
    ind_maxima_right = len(theta_wrap)-1
    ind_maxima_left  = 0
    
    #_________________________________________________________________________________________________________________________________________
    # < theta_central / index_central > are the angle/index corresponding to the STRONGEST MINIMA in the wrapped theta array (theta_wrap)
    index_central,theta_central = -1,  -1
    
    if theta_guess == 'minima':
        if int(index_strongest_min + len(theta_smooth)) > 0 and int(index_strongest_min + len(theta_smooth)) < len(theta_wrap):
            index_central = int(index_strongest_min + len(theta_smooth)) 
            theta_central = theta_wrap[index_central]
            
    elif theta_guess == 'previous':
        if theta_prev != -1:
            index_central = -1
            theta_central = theta_prev                                   
            for j in range(len(theta_wrap)):
                if int(theta_wrap[j]*180/numpy.pi) == int(theta_prev*180/numpy.pi):               
                    index_central = j 
                    break
            if index_central == -1:
                for j in range(len(theta_wrap)):
                    if int(theta_wrap[j]*180/numpy.pi)-1 == int(theta_prev*180/numpy.pi):         
                        index_central = j-1 
                        break
                    if int(theta_wrap[j]*180/numpy.pi)+1 == int(theta_prev*180/numpy.pi):         
                        index_central = j+1 
                        break
        else:
            print('Error -  Must imput a value for theta_prev if keyword argument <theta_guess> is set to <previous>')

    else:
        print('Invalid keyword argument <theta_guess> --- must be <minima> or <previous>')
    
    # < ind_maxima_right > is the index of the maximum NEAREST to the STRONGEST MINIMA, to the right (higher theta), in the WRAPPED theta array
    for i in range(index_central,len(theta_wrap)):
        if i in ind_maxima_cutoff_wrap:
            ind_maxima_right = i
        if ind_maxima_right == 471:
            break
    # < ind_maxima_left > is the index of the maximum NEAREST to the STRONGEST MINIMA, to the left (lower theta), in the WRAPPED theta array
    for i in range(index_central,0,-1):
        if i in  ind_maxima_cutoff_wrap:
            ind_maxima_left = i
            break

    #_________________________________________________________________________________________________________________________________________

    theta_right0 = theta_wrap[int(ind_maxima_right)]*180/numpy.pi                     # Theta corresponding to the maximum to the right of the STRONGEST min.
    theta_left0  = theta_wrap[int(ind_maxima_left)] *180/numpy.pi                     # Theta corresponding to the maximum to the left of the STRONGEST min.

    interval_left  = (theta_central*180/numpy.pi - theta_left0)                      # Theta interval to the right, to the nearest maximum / in degrees
    interval_right = (theta_right0   - theta_central*180/numpy.pi)                   # Theta interval to the left , to the nearest maximum / in degrees
    
    # If the interval to the nearest maxima to the right/left is larger than <bound_theta> deg, set this interval equal to bound_theta deg.
    ind_left, ind_right = -1, -1 
    
    
    if interval_left > bound_theta or interval_left < 15:                                                    # The fiting interval in theta cannot be larger than < bound_theta (30/45 deg) > or smaller than 15 deg
        for i in range(index_central,0,-1):
            if int (abs(theta_wrap[index_central]-theta_wrap[i])*180/numpy.pi) == (bound_theta):
                ind_left      = i                                                                            # Index of the theta value <bound_theta> deg below the strongest minima
                theta_left0    = theta_wrap[i]*180/numpy.pi                                                  # Theta value <bound_theta> deg below the strongest minima
                interval_left = abs(theta_wrap[index_central]*180/numpy.pi - theta_left0)                    # ~ <bound_theta> deg
                break
        if ind_left == -1:
            for i in range(index_central,0,-1):
                if int (abs(theta_wrap[index_central]-theta_wrap[i])*180/numpy.pi) ==( bound_theta - 1 ):
                    ind_left      = i-1                                                             # Index of the theta value <bound_theta - 1> deg below the strongest minima
                    theta_left0   = theta_wrap[i-1]*180/numpy.pi                                    # Theta value <bound_theta - 1> deg below the strongest minima
                    interval_left = abs(theta_wrap[index_central]*180/numpy.pi - theta_left0)       # ~ <bound_theta - 1> deg
                    break
                if int (abs(theta_wrap[index_central]-theta_wrap[i])*180/numpy.pi) ==( bound_theta + 1 ):
                    ind_left      = i+1                                                             # Index of the theta value <bound_theta + 1> deg below the strongest minima
                    theta_left0   = theta_wrap[i+1]*180/numpy.pi                                    # Theta value <bound_theta + 1> deg below the strongest minima
                    interval_left = abs(theta_wrap[index_central]*180/numpy.pi - theta_left0)       # ~ <bound_theta + 1> deg
                    break
    else: 
        ind_left = ind_maxima_left
    if interval_right > bound_theta or interval_right < 15:                                    # The fiting interval in theta cannot be larger than < bound_theta (30/45 deg) > or smaller than 15 deg
        for i in range(index_central,len(theta_wrap)):
            if int (abs(theta_wrap[i] - theta_wrap[index_central])*180/numpy.pi) == (bound_theta) :
                ind_right      = i                                                             # Index of the theta value <bound_theta> deg above the strongest minima
                theta_right0   = theta_wrap[i]*180/numpy.pi                                    # Theta value <bound_theta> deg above the strongest minima
                interval_right = abs(theta_right0 - theta_wrap[index_central]*180/numpy.pi)    # ~ <bound_theta> deg
                break
        if ind_right == -1:
            for i in range(index_central,len(theta_wrap)):
                if int (abs(theta_wrap[i] - theta_wrap[index_central])*180/numpy.pi) == ( bound_theta - 1 ):
                    ind_right      = i-1                                                            # Index of the theta value <bound_theta - 1> deg above the strongest minima
                    theta_right0    = theta_wrap[i-1]*180/numpy.pi                                  # Theta value <bound_theta - 1> deg above the strongest minima
                    interval_right = abs(theta_right0 - theta_wrap[index_central]*180/numpy.pi)     # ~ <bound_theta - 1> deg
                    break
                if int (abs(theta_wrap[i] - theta_wrap[index_central])*180/numpy.pi) == ( bound_theta + 1):
                    ind_right      = i+1                                                           # Index of the theta value <bound_theta + 1> deg above the strongest minima
                    theta_right0    = theta_wrap[i+1]*180/numpy.pi                                  # Theta value <bound_theta + 1> deg above the strongest minima
                    interval_right = abs(theta_right0 - theta_wrap[index_central]*180/numpy.pi)     # ~ <bound_theta + 1> deg
                    break
    else:
        ind_right = ind_maxima_right

    ' 7) Finally - Fit a von Mises function over the theta interval defined above:'
    #### Consider the case where ind_left = ind_right = -1 OR ind_left > ind_right
    if ind_left != -1 and ind_right != -1 and ind_left < ind_right:
        fitting_theta = theta_wrap[ind_left:ind_right]  # Interval in theta over which to fit the von Mises function --- as defined above
        fitting_r_AB  = rd_AB_wrap[ind_left:ind_right]  # Interval in  R_AB over which to fit the von Mises function --- as defined above

        # Initial guess parameters for the von Mises function #
        #________________________________________________________
        a0  = -460
        k0  =  5.22
        mu0 = theta_central
        b0  = 478
        sl0 = 1 
        guess  = numpy.array([k0,a0,mu0,b0,sl0])
        #________________________________________________________
        # Define lower bounds for the fitting parameters #
        #________________________________________________________
        if theta_central >= bound_theta*numpy.pi/180:
            theta_lower_bound = theta_central - bound_theta*numpy.pi/180
        else:
            theta_lower_bound = 0

        if theta_central <= (180-bound_theta)*numpy.pi/180:
            theta_upper_bound = theta_central + bound_theta*numpy.pi/180
        else:
            theta_upper_bound = numpy.pi

        bounds  = []
        bounds1 = numpy.array([ 0,    -5000,      theta_lower_bound,     -200,  -1.5  ])
        bounds2 = numpy.array([ 10,    -50,       theta_upper_bound,      700,   1.5  ])
        bounds.append(bounds1)
        bounds.append(bounds2)
        #________________________________________________________

        #________________________________________________________
        if error == numpy.array([-1]):
            error = numpy.zeros(len(fitting_theta)) + 1
        #________________________________________________________
        # Extract best-fitting parameters 
        try:
            array, covar = optimise_vm(fitting_theta,fitting_r_AB,guess,error,bounds)
            k,  a , mu, b, sl = array[0],   array[1],   array[2],    array[3],    array[4] 
        except:
            print('Fitting of Von Mises function failed')
            k,  a , mu, b, sl = -9999, -9999, 0, -9999, -9999
            array = numpy.array([-1])
            covar = numpy.array([-1])

        
        vonMises0 =  von_mises(fitting_theta,k0,a0,mu0,b0,sl0)  # von Mises function for the initial guess parameters
        vonMises  =  von_mises(fitting_theta,k,a,mu,b,sl)       # von Mises function for the best-fitting  parameters
        return mu, array, covar, vonMises,              theta_smooth, rd_AB_smooth,               ind_minima_cutoff, ind_maxima_cutoff,        theta_left0, theta_right0,         fitting_theta, grad, number_minima_after, number_maxima_after,         cutoff_plot
    
    else:
        print('No data found left/right of the best estimate of theta:')
        print('Values returned for left/rignt indicies are:')
        print('Left index = ',ind_left)
        print('Right index = ',ind_right)
        
        return -1, numpy.array([-1,-1,-1,-1,-1]),numpy.array([-1,-1,-1,-1,-1]), numpy.array([-1]),           theta_smooth, rd_AB_smooth,         ind_minima_cutoff, ind_maxima_cutoff,        theta_left0, theta_right0,          numpy.array([-1]), grad,  number_minima_after, number_maxima_after,      cutoff_plot

#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________

def extrema(y,kw,cutoff_extrema):
    #_______________________________________________
    #        y        - 1D array for which extrema (local/global maxima and minima are computed)
    #                   The function uses the numpy.diff function to compute the 1st order differential of a 1D function
    #        kw       - 'max' or 'min', tels the function to compute either maxima or minima
    # cutoff_extrema  - between [0,1] - eliminate all maxima/minima with a weight smaller than this number times the weight of the strongest maxima/minima
    #_______________________________________________
    grad           = numpy.diff(y,n=1)
    #__________________
    ind_minima     = []
    minima         = []
    peak_width_min = []
    weight_min     = []
    #__________________
    ind_maxima     = []
    maxima         = []
    peak_width_max = []
    weight_max     = []
    #__________________
    

    for i in range(len(grad)-1):
        
        sum_within_width_min = 0
        left_width_min       = 0
        right_width_min      = 0

        'Minima:'
        '_________________________________________________________________________'
        '_________________________________________________________________________'
        if (grad[i]<0 or grad[i]==0) and grad[i+1] > 0 :
            chosen_ind_min = i
            if i > 0 and abs(grad[i-1]) < abs(grad[chosen_ind_min]):
                chosen_ind_min = i-1
            if i < len(grad)-1 and abs(grad[i+1]) < abs(grad[chosen_ind_min]):
                chosen_ind_min = i+1
            
            ind_minima.append(chosen_ind_min)
            minima.append(grad[chosen_ind_min])

            # Left Width #
            #_____________________________
            if chosen_ind_min > 0:  
                for j in range(chosen_ind_min-1,0,-1):
                    if grad[j] > 0:
                        left_width_min = j + 1
                        break
                    else:
                        left_width_min = j
                        sum_within_width_min = sum_within_width_min + abs(grad[j])
            else:
                left_width_min = 0   

            # Right Width # 
            #_____________________________
            if chosen_ind_min < len(grad)-1:
                for j in range(chosen_ind_min+1,len(grad)-1):
                    if grad[j] < 0:
                        right_width_min = j - 1 
                        break
                    else:
                        right_width_min = j
                        sum_within_width_min = sum_within_width_min + abs(grad[j])

            else:
                right_width_min = len(grad) - 1
            
            pw = right_width_min - left_width_min
            peak_width_min.append(pw)
            weight_min.append(sum_within_width_min)

        sum_within_width_max = 0
        left_width_max       = 0
        right_width_max      = 0
        
        'Maxima:'
        '_________________________________________________________________________'
        '_________________________________________________________________________'
        if (grad[i]>0 or grad[i]==0) and grad[i+1] < 0 :
            chosen_ind_max = i
            if i > 0 and abs(grad[i-1]) < abs(grad[chosen_ind_max]):
                chosen_ind_max = i-1
            if i < len(grad)-1 and abs(grad[i+1]) < abs(grad[chosen_ind_max]):
                chosen_ind_max = i+1

            ind_maxima.append(chosen_ind_max)
            maxima.append(grad[chosen_ind_max])

            # Left Width #
            #_____________________________
            if chosen_ind_max > 0:  
                for j in range(chosen_ind_max-1,0,-1):
                    if grad[j] < 0:
                        left_width_max = j + 1
                        break
                    else:
                        left_width_max = j
                        sum_within_width_max = sum_within_width_max + abs(grad[j])
            else:
                left_width_max = 0   

            # Right Width # 
            #_____________________________
            if chosen_ind_max < len(grad)-1:
                for j in range(chosen_ind_max+1,len(grad)-1):
                    if grad[j] > 0:
                        right_width_max = j - 1 
                        break
                    else:
                        right_width_max = j
                        sum_within_width_max = sum_within_width_max + abs(grad[j])

            else:
                right_width_max = len(grad) - 1
            
            pw = right_width_max - left_width_max
            peak_width_max.append(pw)
            weight_max.append(sum_within_width_max)

    ind_minima      = numpy.array(ind_minima)
    minima          = numpy.array(minima)
    peak_width_min  = numpy.array(peak_width_min)
    weight_min      = numpy.array(weight_min)
    if len(weight_min)!=0:
        norm_weight_min = weight_min/numpy.nanmax(weight_min) 
    else:
        norm_weight_min = numpy.array([])

    ind_maxima      = numpy.array(ind_maxima)
    maxima          = numpy.array(maxima)
    peak_width_max  = numpy.array(peak_width_max)
    weight_max      = numpy.array(weight_max)
    if len(weight_max)!=0:
        norm_weight_max = weight_max/numpy.nanmax(weight_max) 
    else:
        norm_weight_max = numpy.array([]) 

    #_____________________________________________________________________________________
    #                                     Test cutoff values
    #_____________________________________________________________________________________

    cutoff     = numpy.linspace(0,1,num=100,endpoint=False)
    num_minima          = len(minima)
    num_maxima          = len(maxima)
    number_minima_after = []
    number_maxima_after = []
    for i in range(len(cutoff)):
        subtract_min,subtract_max  = 0,0
        cf = cutoff[i]
        for j in range(len(norm_weight_min)):
            if norm_weight_min[j] < cf:
                subtract_min = subtract_min + 1
        for j in range(len(norm_weight_max)):
            if norm_weight_max[j] < cf:
                subtract_max = subtract_max + 1
        number_minima_after.append( num_minima - subtract_min )
        number_maxima_after.append( num_maxima - subtract_max )
    number_minima_after = numpy.array(number_minima_after)
    number_maxima_after = numpy.array(number_maxima_after)
    
    # cutoff_extrema # 
    minima_cutoff, ind_minima_cutoff, norm_weight_min_cutoff = [],[],[]
    maxima_cutoff, ind_maxima_cutoff, norm_weight_max_cutoff = [],[],[]
    for j in range(len(norm_weight_min)):
        if norm_weight_min[j] >= cutoff_extrema:
            minima_cutoff.append(minima[j])
            ind_minima_cutoff.append(ind_minima[j])      
            norm_weight_min_cutoff.append(norm_weight_min[j])
    for j in range(len(norm_weight_max)):
        if norm_weight_max[j] >= cutoff_extrema:
            maxima_cutoff.append(maxima[j])
            ind_maxima_cutoff.append(ind_maxima[j]) 
            norm_weight_max_cutoff.append(norm_weight_max[j])
    
    minima_cutoff, ind_minima_cutoff, norm_weight_min_cutoff = numpy.array(minima_cutoff),  numpy.array(ind_minima_cutoff),  numpy.array(norm_weight_min_cutoff)
    maxima_cutoff, ind_maxima_cutoff, norm_weight_max_cutoff = numpy.array(maxima_cutoff),  numpy.array(ind_maxima_cutoff),  numpy.array(norm_weight_max_cutoff)

    if kw == 'min':
        return grad, ind_minima, minima, peak_width_min, weight_min, norm_weight_min, cutoff, number_minima_after,       minima_cutoff,  ind_minima_cutoff, norm_weight_min_cutoff
    if kw == 'max':
        return grad, ind_maxima, maxima, peak_width_max, weight_max, norm_weight_max, cutoff, number_maxima_after,       maxima_cutoff,  ind_maxima_cutoff, norm_weight_max_cutoff

#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________


def radon(velmap,dx,dy,pixelres,r_app,gal_ind, plot='no' , plot_at_rho = 3.55,component='stars'):   # xmin,ymin,xmax,ymax
    #__________________________________________________________________________________
    
    # NOTE: If velmap is N x M pixels, the code is computed to work for x_min/max = +/- (N-1)/2 | y_min/max = +/- (M-1)/2 
    #__________________________________________________________________________________
    #      xmin  : x - coordinate of the leftmost edge of the velocity map  (!! IMPUT xmin = -1 if not known !!)
    #      ymin  : y - coordinate of the bottom edge of the velocity map    (!! IMPUT xmin = -1 if not known !!)
    #      xmax  : x - coordinate of the rightmost edge of the velocity map  (!! IMPUT xmin = -1 if not known !!)
    #      ymax  : y - coordinate of the top edge of the velocity map (!! IMPUT xmin = -1 if not known !!)
    #        dx  : Step size in the x-direction (in # of pixels --- to convert into arcsec, mulyiply by pixelres)
    #        dy  : Step size in the y-direction (in # of pixels --- to convert into arcsec, mulyiply by pixelres)
    #
    #    velmap  : The 2D velocity field for which Radon Statistics are computed
    #  pixelres  : Resolution of pixels in the velocity maps
    #    r_app   : Radon apperture used to compute BOUNDED Radon transforms | in # of pixels
    #_________________________________________________________________________________________
    #
    # 
    #
    #_________________________________________________________________________________________
    
    x_dim        = len(velmap[:,0])   # = 50 for SAMI
    y_dim        = len(velmap[0,:])   # = 50 for SAMI
    x_dim_arcsec = x_dim*pixelres     # = 25'' for SAMI
    y_dim_arcsec = y_dim*pixelres     # = 25'' for SAMI

    #____________________________________________________________________________________________
    #                                EXAMPLES FOR the SAMI survey
    #___________________________________________________________________________________________
    'Approach using x_ind has been discontinued for time purposes'
    #             x_ind:                 |                  x_ind_arcsec      
    #[[ 0.  0.  0. ...  0.  0.  0.]      |   [[-12.5 -12.5 -12.5 ... -12.5 -12.5 -12.5]     |         |
    #[ 1.  1.  1. ...  1.  1.  1.]       |   [-12.  -12.  -12.  ... -12.  -12.  -12. ]      |         |
    #[ 2.  2.  2. ...  2.  2.  2.]       |   [-11.5 -11.5 -11.5 ... -11.5 -11.5 -11.5]      |      X  |
    #...                                 |   ...                                            |         |
    #[47. 47. 47. ... 47. 47. 47.]       |   [ 11.   11.   11.  ...  11.   11.   11. ]      |         | ____________>
    #[48. 48. 48. ... 48. 48. 48.]       |   [ 11.5  11.5  11.5 ...  11.5  11.5  11.5]      |        /|/      
    #[49. 49. 49. ... 49. 49. 49.]]      |   [ 12.   12.   12.  ...  12.   12.   12. ]]     |                Y

    #             y_ind:                 |                  y_ind_arcsec                    |
    #[[ 0.  1.  2. ... 47. 48. 49.]      |  [[-12.5 -12.  -11.5 ...  11.   11.5  12. ]      |
    #[ 0.  1.  2. ... 47. 48. 49.]       |  [-12.5 -12.  -11.5 ...  11.   11.5  12. ]       |
    #...                                 |   ...                                            |
    #[ 0.  1.  2. ... 47. 48. 49.]       |  [-12.5 -12.  -11.5 ...  11.   11.5  12. ]       |
    #[ 0.  1.  2. ... 47. 48. 49.]]      |  [-12.5 -12.  -11.5 ...  11.   11.5  12. ]]      |
    #______________________________________________________________________________________________
 
    # NOTE: THIS IS THE RIGHT WAY OF CALCULATING XMIN / XMAX
    ymin = -(y_dim - 1)/2 
    ymax = (y_dim - 1)/2
    xmin = -(x_dim - 1)/2 
    xmax = (x_dim - 1)/2

    ymin_arcsec = -(y_dim*pixelres - pixelres)/2 
    ymax_arcsec = (y_dim*pixelres - pixelres)/2
    xmin_arcsec = -(x_dim*pixelres - pixelres)/2 
    xmax_arcsec = (x_dim*pixelres - pixelres)/2
    
    #______________________________________________________________________________________________
    theta_min,  theta_max = 0,180
    n_theta = int(round(numpy.pi*numpy.sqrt(0.5*(x_dim**2 + y_dim**2)))  )
    dtheta = (theta_max-theta_min)/n_theta
    #____ARRAYS_____
    theta_arr    = numpy.linspace(0,n_theta, num=n_theta, endpoint = False)*dtheta   # in degrees
    theta_arr    = theta_arr*(numpy.pi/180)                                          # in arcsec
    
    cos_theta    = numpy.cos(theta_arr)
    sin_theta    = numpy.sin(theta_arr)
    sin_theta[0] = sin_theta[1]

    drho        = 0.5*numpy.sqrt(dx**2 + dy**2) 
    drho_arcsec = 0.5*numpy.sqrt((dx*pixelres)**2 + (dy*pixelres)**2)
    n_rho       = int(2*numpy.sqrt(ymax**2 + xmax**2)/drho)                       
    rmin        = -0.5*(n_rho-1)*drho                                            
    rmin_arcsec = -0.5*(n_rho-1)*drho_arcsec  
    #____ARRAYS_____
    
    rho_arr        = numpy.linspace(0,n_rho, num = n_rho, endpoint = False)*drho + rmin 
    rho_arr_arcsec = numpy.linspace(0,n_rho, num = n_rho, endpoint = False)*drho_arcsec + rmin_arcsec 
    
    #______________________________________________
    #               Create Radon Image
    #______________________________________________
    rd    = numpy.zeros([n_rho, n_theta]) # [n_rho, n_theta]
    rd_A  = numpy.zeros([n_rho, n_theta])
    rd_AB = numpy.zeros([n_rho, n_theta])

    diff   = rd
    vmax   = rd
    vmin   = rd
    length = rd 
    
    maskfrac = length*0+1
    edge     = maskfrac*0+999
    
    #___ IF errors are to be calculated - use these_____
    dtim = rd
    ddiff = rd

    inds_rd     = numpy.zeros([n_theta, n_rho]) 
    inds_map    = numpy.zeros([x_dim,y_dim])  
    weights_arr = numpy.zeros([x_dim*y_dim,n_theta*n_rho])


    #________ CALCULATE THE VARIABLES FOR THE PATHS (equations 4-7) in Stark et al. (2018)______
    
    a           = -(dx/dy)*cos_theta/sin_theta 
    '-------------------------'
    b_ct          = xmin*cos_theta + ymin*sin_theta
    
    b_norm        = dy*sin_theta

    #arg_sin_below = numpy.argwhere(sin_theta <= numpy.sqrt(2)/2)
    
    for i in range(len(a)):
        if abs(sin_theta[i]) <= numpy.sqrt(2)/2 :
            a[i]              = 1/a[i]
            b_norm[i]         = dx*cos_theta[i] 

    xinds        = numpy.linspace(0,x_dim,num=x_dim,endpoint=False)    #  xinds = numpy.linspace(0,x_dim,num=x_dim,endpoint=False)     # GO BETWEEN xmin --- xmax instead of 0 --- x_dim !!!!!!!
    yinds        = numpy.linspace(0,y_dim,num=y_dim,endpoint=False)
    
    xind0        = max(xinds)/2
    yind0        = max(yinds)/2

    #____________BEGIN THE SUMMATION IN EQUATION (3) OF Stark et al. 2018 TO COMPUTE THE RADON TRANSFORM___________
    
    #velmap_copy = numpy.copy(velmap) 
    
    mask_radon_AB = numpy.zeros([n_rho, n_theta]) + 1 # Used to mask regions in the RADON map which have used masked spaxels or spaxels at the edge of the vel. map

    print('TIME BEFORE THE CONSTRUCTION OF RADON MAPS:')
    print("--- %s seconds ---" % (time.time() - start_time))
    for i in range(0, n_theta):
        #__________________________________________________
        if abs(sin_theta[i] > numpy.sqrt(2)/2):
            path              = 1
            pre_factor        = dx/abs(sin_theta[i]) 
        else:
            path              = 0
            pre_factor        = dy/abs(cos_theta[i]) 
        b        = (rho_arr - b_ct[i])/b_norm[i]                       # /in pixels # This is an array - for the given theta AND all possible rhos
        #__________________________________________________
        
        for j in range (0, n_rho):
            ' Construct the integration path for the given rho & theta:'
            # NOTE: a is a function of ONLY theta so we choose a[i] as i is the theta index ||| b is an array calculated for all rhos and for theta[i], so we choose b[j]
            # (See below)
            if path == 1:
                
                ix        = xinds
                iy        = numpy.round(a[i]*xinds + b[j])    
                ind_iy1     =  numpy.argwhere(iy >= y_dim) # 50
                ind_iy2     =  numpy.argwhere(iy <= -y_dim)
                ind_iy3     =  numpy.argwhere(iy < 0)
                ind_iy4     =  numpy.argwhere(iy > -y_dim)
                ind_iy34    =  numpy.intersect1d(ind_iy3, ind_iy4) 
                iy[ind_iy1] = numpy.nan
                iy[ind_iy2] = numpy.nan
                iy[ind_iy34] = numpy.nan
                
            else:
                
                ix        = numpy.round(a[i]*yinds + b[j])
                iy        = yinds
                ind_ix1     =  numpy.argwhere(ix >= x_dim)
                ind_ix2     =  numpy.argwhere(ix <= -x_dim)
                ind_ix3     =  numpy.argwhere(ix < 0)
                ind_ix4     =  numpy.argwhere(ix > -x_dim)
                ind_ix34    =  numpy.intersect1d(ind_ix3, ind_ix4) 
                ix[ind_ix1] = numpy.nan
                ix[ind_ix2] = numpy.nan
                ix[ind_ix34] = numpy.nan
                
            xcent        = rho_arr[j]*cos_theta[i]
            ycent        = rho_arr[j]*sin_theta[i]
            #________________________________________
            rad          = numpy.zeros(len(ix))
            
            ix_notnan            = numpy.argwhere(numpy.isnan(ix)==False)
            iy_notnan            = numpy.argwhere(numpy.isnan(iy)==False)
            args_ixy_notnan      = numpy.intersect1d(ix_notnan, iy_notnan) 
            rad[args_ixy_notnan] = numpy.sqrt(((ix[args_ixy_notnan]-xind0)-xcent)**2 + ((iy[args_ixy_notnan]-yind0)-ycent)**2)
            
            ix_positive        = numpy.argwhere(ix > 0)
            ix_within_dim      = numpy.argwhere(ix < x_dim)
            rad_within_app     = numpy.argwhere(rad < r_app)
            iy_positive        = numpy.argwhere(iy > 0)
            iy_within_dim      = numpy.argwhere(iy < y_dim)
            #---------------      -----------------      ---------------
            int1           = numpy.intersect1d(ix_positive, ix_within_dim) 
            int2           = numpy.intersect1d(int1, rad_within_app) 
            int3           = numpy.intersect1d(int2, iy_positive) 
            int4           = numpy.intersect1d(int3, iy_within_dim) 

            ix_cut = ix[int4]
            iy_cut = iy[int4]
            
            # 
            #____________________________________________________________________________________________________________
            # Flag regions in R_AB where the lines overlap --- MASKED spaxels --- OR --- the edge of the vel. field ---
            #____________________________________________________________________________________________________________
            
            
            # Masking criterion 1 - Using a mask for the vel. map #
            '''
            for k in range(len(ix)):
                if mask[iy[k],ix[k]] == 0:
                    mask_radon_AB[j,i] = 0
            '''
            # Masking criterion 2 - If the spaxels at [ix_cut | iy_cut] are at the edge of the vel. map #
            #count_nans = 0
            for k in range(len(ix_cut)):
                count_nans = 0
                for l in range(int(ix_cut[k]-1),int(ix_cut[k]+2)):
                    for m in range(int(iy_cut[k]-1),int(iy_cut[k]+2)):
                        if l < x_dim and m < y_dim and numpy.isnan(velmap[m,l]) == True:
                            count_nans = count_nans + 1
                if count_nans >=2:
                    mask_radon_AB [j,i] = 0
                
            #____________________________________________________________________________________________________________
            #____________________________________________________________________________________________________________

            vel_radon    = []
            vel_radon_AB = []
            
            'Use to calculate R / R_A:'
            if len(ix)>0 and len(iy)>0:
                for k in range(len(ix)):
                    if  numpy.isnan(ix[k])==False and numpy.isnan(iy[k])==False:
                        if numpy.isnan(velmap[int(iy[k]),int(ix[k])]) == False:
                            vel_radon.append(velmap[int(iy[k]),int(ix[k])])
                
            else: 
                vel_radon = numpy.array([0])
            
            vel_radon = numpy.array(vel_radon) 
            
            'Use to calculate R_AB:'
            if len(ix_cut)>0 and len(iy_cut)>0:
                for k in range(len(ix_cut)):
                    if  numpy.isnan(ix_cut[k])==False and numpy.isnan(iy_cut[k])==False:
                        if numpy.isnan(velmap[int(iy_cut[k]),int(ix_cut[k])]) == False:
                            vel_radon_AB.append(velmap[int(iy_cut[k]),int(ix_cut[k])])
            else: 
                vel_radon_AB = numpy.array([0])
            
            vel_radon_AB = numpy.array(vel_radon_AB) 
    
            #_______________________________________________________________________________
            # HIGHLIGHT SPAXELS FOR A GIVEN (theta / rho) COMBINATION:
            #_______________________________________________________________________________
            #_______________________________________________________________________________
            
            'Discontinued to save time - see Radon 5.0'

            #_______________________________________________________________________________
            #_______________________________________________________________________________

            sum_rd           = 0
            sum_rd_A         = 0 
            sum_rd_AB        = 0
            med_vel_radon    = numpy.nanmedian(vel_radon)   # Try average too
            med_vel_radon_AB = numpy.nanmedian(vel_radon_AB)
            
            sum_rd        =  pre_factor*numpy.nansum(vel_radon)
            sum_rd_A      =  pre_factor*numpy.nansum(abs(vel_radon - med_vel_radon)) 
            sum_rd_AB     =  pre_factor*numpy.nansum(abs(vel_radon_AB - med_vel_radon_AB)) 
        
            
            rd [j,i]    = sum_rd
            rd_A [j,i]  = sum_rd_A
            rd_AB [j,i] = sum_rd_AB

            max_rd_A  = numpy.nanmax(rd_A)
            max_rd_AB = numpy.nanmax(rd_AB)

            rd_A_scaled  =  numpy.copy(rd_A)
            if max_rd_A!=0:
                rd_A_scaled  = rd_A/max_rd_A
            
            rd_AB_scaled =  numpy.copy(rd_AB)
            if max_rd_AB!=0:
                rd_AB_scaled  = rd_AB/max_rd_AB

    print('TIME AFTER RADON MAPS HAVE BEEN COMPUTED:')
    print("--- %s seconds ---" % (time.time() - start_time))
            
    #_________________________________________________________________________________________________________________________________________________
    #                                                          Radon Profile extraction
    #_________________________________________________________________________________________________________________________________________________
    
    # First mask the regions in the BOUNDED Radon prifiles using mask_radon_AB
    
    #rd_AB        = rd_AB * mask_radon_AB
    #rd_AB_scaled = rd_AB_scaled * mask_radon_AB
    

    ' 1) Take a cut through R_AB[rho,theta] at rho = 0'
    ind0 = numpy.where(rho_arr >= 0)
    ind0p = ind0[0][0]                  # Index of the closest POSITIVE value to zero  --- Use this 
    ind0m = ind0p - 1                   # Index of the closest NEGATIVE value to zero 
    
    rd_AB_0 = rd_AB[ind0p,:]            # Take the one on the positive side
    
    cf = 0.33 # 0.02
    mu0,array0,covar0,vonMises0, theta_smooth0, rd_AB_smooth0, ind_minima_cutoff0, ind_maxima_cutoff0, theta_left0, theta_right0, fitting_theta0, grad0, number_minima_after0, number_maxima_after0, cutoff0          =         radon_slice_fit(rd_AB_0, theta_arr, cf , bound_theta=45, theta_guess = 'minima', theta_prev = -1,error = numpy.array([-1]), rho=rho_arr[ind0p])
    print('TIME AFTER FITTING SLICE THROUGH RADON PROFILE AT rho=0:')
    print("--- %s seconds ---" % (time.time() - start_time))
    theta_hat = numpy.zeros(n_rho)-9999   # ! in RADIANS ! 
    theta_hat[ind0p] = mu0


    ' 8) Now repeat the steps 1-7 above for all NEGATIVE rho:  --- See steps above for comments on the working process'
    # NOTE: Initial guess in theta for the von Mises function is the besf-fitting value for the previous rho. See Stark et al. (2018) for more details
    #_________________________________________________________________________________________________________________________________________________
    #_________________________________________________________________________________________________________________________________________________
    
    rho_x     = abs(rho_arr - plot_at_rho/0.5)   # NOTE: rho_arr is in pixels & plot_at_rho is in arcsec
    min_rho_x = numpy.min(rho_x) 
    index_x   =  numpy.where(rho_x == min_rho_x)
    

    # Below rho = 0 (rho < 0):
    #____________________________
    #____________________________
    for i in range(ind0m,0,-1):                   
        rd_AB_i  = rd_AB[i,:]   
        nz_rd_AB =  numpy.count_nonzero(rd_AB_i)
        if len(rd_AB_i)==len(theta_arr) and nz_rd_AB >=90:
            if  theta_hat[i+1]!=-9999 and theta_hat[i+1]!=-1:
                mu,array,covar,vonMises, theta_smooth, rd_AB_smooth, ind_minima_cutoff_i, ind_maxima_cutoff_i, theta_left, theta_right, fitting_theta_i, grad, number_minima_after_i, number_maxima_after_i, cutoff_i          =         radon_slice_fit(rd_AB_i, theta_arr, cf , bound_theta=30, theta_guess = 'previous', theta_prev = theta_hat[i+1],error = numpy.array([-1]), rho=rho_arr[i])
            else:
                mu,array,covar,vonMises, theta_smooth, rd_AB_smooth, ind_minima_cutoff_i, ind_maxima_cutoff_i, theta_left, theta_right, fitting_theta_i, grad, number_minima_after_i, number_maxima_after_i, cutoff_i          =         radon_slice_fit(rd_AB_i, theta_arr, cf , bound_theta=30, theta_guess = 'minima',error = numpy.array([-1]), rho=rho_arr[i])
            theta_hat[i] = mu
        else:
            mu, vonMises, theta_smooth, rd_AB_smooth, ind_minima_cutoff_i, ind_maxima_cutoff_i, fitting_theta_i,   theta_left, theta_right     =  0,   numpy.array([-1]),  numpy.array([-1])  ,  numpy.array([-1])  ,  numpy.array([0])  ,  numpy.array([0]), numpy.array([0])   , -1, -1
        if i == index_x[0][0]:
            mu_plot, vonMises_plot, theta_smooth_plot, rd_AB_smooth_plot, ind_minima_cutoff_plot, ind_maxima_cutoff_plot, fitting_theta_plot, theta_left_plot, theta_right_plot     =   mu,   vonMises, theta_smooth, rd_AB_smooth, ind_minima_cutoff_i, ind_maxima_cutoff_i, fitting_theta_i,  theta_left,  theta_right

    print('TIME AFTER FITTING ALL RADON MAP AT NEGATIVE rho:')
    print("--- %s seconds ---" % (time.time() - start_time))
    # Above rho = 0 (rho < 0):
    #____________________________
    #____________________________
    
    for i in range(ind0p+1,n_rho):                      
        rd_AB_i = rd_AB[i,:]   
        nz_rd_AB =  numpy.count_nonzero(rd_AB_i)
        #print('rd_AB_i:')
        #print(rd_AB_i)
        if len(rd_AB_i)==len(theta_arr) and nz_rd_AB >=90:
            if  theta_hat[i-1]!=0 and theta_hat[i-1]!=-1:
                mu,array,covar,vonMises, theta_smooth, rd_AB_smooth, ind_minima_cutoff_i, ind_maxima_cutoff_i, theta_left, theta_right, fitting_theta_i, grad, number_minima_after_i, number_maxima_after_i, cutoff_i          =         radon_slice_fit(rd_AB_i, theta_arr, cf , bound_theta=30, theta_guess = 'previous', theta_prev = theta_hat[i-1],error = numpy.array([-1]), rho=rho_arr[i])
            else:
                mu,array,covar,vonMises, theta_smooth, rd_AB_smooth, ind_minima_cutoff_i, ind_maxima_cutoff_i, theta_left, theta_right, fitting_theta_i, grad, number_minima_after_i, number_maxima_after_i, cutoff_i          =         radon_slice_fit(rd_AB_i, theta_arr, cf , bound_theta=30, theta_guess = 'minima',error = numpy.array([-1]), rho=rho_arr[i])
            theta_hat[i] = mu
        else:
            mu,  vonMises, theta_smooth, rd_AB_smooth, ind_minima_cutoff_i, ind_maxima_cutoff_i, fitting_theta_i, theta_left, theta_right     = 0, numpy.array([-1]),  numpy.array([-1])  ,  numpy.array([-1])  ,  numpy.array([0])  ,  numpy.array([0]),   numpy.array([0]),  -1,   -1
        if i == index_x[0][0]:
            mu_plot,  vonMises_plot, theta_smooth_plot, rd_AB_smooth_plot, ind_minima_cutoff_plot, ind_maxima_cutoff_plot, fitting_theta_plot , theta_left_plot, theta_right_plot    =    mu  ,vonMises, theta_smooth, rd_AB_smooth, ind_minima_cutoff_i, ind_maxima_cutoff_i,  fitting_theta_i, theta_left,  theta_right

    print('=========================================================')
    print('=========================================================')
    print('=========================================================')

    print('TIME AFTER FITTING ALL RADON MAP AT POSSITIVE rho:')
    print("--- %s seconds ---" % (time.time() - start_time))

    theta_hat = theta_hat *180/numpy.pi
    
    #____________________________________________
    #          Radon Profile extraction
    #____________________________________________
    
    theta_hat_rp = []
    rho_arr_rp   = []
    Nij          = 0
    
    theta_hat_rp = theta_hat[numpy.argwhere(theta_hat >= 0)]
    rho_arr_rp   =  rho_arr[numpy.argwhere(theta_hat >= 0)]
    rho_arr_abs  = abs(rho_arr)

    'Determine the position angle at rho = 0 (take the average of the 2 values closest to zero):'
    index_min  = numpy.argwhere(numpy.around(rho_arr_abs,decimals=3) == numpy.around(numpy.nanmin(rho_arr_abs),decimals=3))
    
    
    try:
        theta_choose = theta_hat[index_min]
        PA = numpy.nansum(theta_choose)/len(theta_choose)
    except:
        PA = -1
    # PA is measured from EAST = 0 counterclockwise - convert this to NORTH = 0
    if PA >= 90:
        PA_N = PA - 90
    elif PA!=-1:
        PA_N = PA + 90
    
    #_______________________________

    'Flip arrays - this will be used in the asymmetry parameter calculation'
    theta_flip = numpy.flip(theta_hat_rp)
    rho_flipped = numpy.flip(-1*(rho_arr_rp))
    
    Nij = len(numpy.intersect1d( numpy.around(rho_arr_rp,decimals=3), numpy.around(rho_flipped,decimals=3)))
    
    mean_theta  = numpy.nanmean(theta_hat_rp)
    stdev_theta = numpy.nanstd(theta_hat_rp)

    'Find redshifted side and define PA_360 from N=0 c.clockwise to REDSHIFTED side:'
    PA_360              = -1
    rot_PA              = rotate(velmap, 90 + PA_N , reshape=False,order=0)
    mean_vm_left        = numpy.nanmean(rot_PA[:,0:int(y_dim/2)]) 
    mean_vm_right       = numpy.nanmean(rot_PA[:,int(y_dim/2):y_dim]) 
    if mean_vm_left > mean_vm_right:
        PA_360 = PA_N + 180
    else:
        PA_360 = PA_N

    if plot == 'yes':
        #_________________________________________________________________________________________________________________________________________________
        #_________________________________________________________________________________________________________________________________________________
        #                                                                   Plots
        #_________________________________________________________________________________________________________________________________________________
        #_________________________________________________________________________________________________________________________________________________

        csfont = {'fontname':'Times New Roman'}

        #_________________________________________________________________________________________________________________________________________________
        #                                             SECTION THROUGH RADON MAP AT ALL Rho (Col. Coded) & at Rho = 0:
        #_________________________________________________________________________________________________________________________________________________
        
        'Discontinued to save execution time - see Radon 5.0'

        #_________________________________________________________________________________________________________________________________________________
        #                                                EXTRA PLOT --- Section through R_AB at Rho = (plot_at_rho):
        #_________________________________________________________________________________________________________________________________________________

        'Discontinued to save execution time - see Radon 5.0'

        #_________________________________________________________________________________________________________________________________________________
        #                                                                 'VEL. MAP' 
        #_________________________________________________________________________________________________________________________________________________
        
        'Discontinued to save execution time - see Radon 5.0'

        #_________________________________________________________________________________________________________________________________________________
        #                                                                  RADON MAPS 
        #_________________________________________________________________________________________________________________________________________________

        rho_min   = numpy.nanmin(rho_arr)
        rho_max   = numpy.nanmax(rho_arr)
        theta_min = numpy.nanmin(theta_arr*180/numpy.pi) 
        theta_max = numpy.nanmax(theta_arr*180/numpy.pi) 

        masked_map1 = numpy.ma.masked_where(rd == 0, rd)
        masked_map2 = numpy.ma.masked_where(rd_A == 0, rd_A)
        masked_map3 = numpy.ma.masked_where(rd_A_scaled == 0, rd_A_scaled)
        masked_map4 = numpy.ma.masked_where(rd_AB == 0, rd_AB)
        masked_map5 = numpy.ma.masked_where(rd_AB_scaled == 0, rd_AB_scaled)


        fig2,axs2 = plt.subplots(ncols=2, nrows=3,figsize=(97,97)) 
        

        plt.subplots_adjust(left=0.052, bottom=0.06, right=0.979, top=0.993, wspace=0, hspace=0.083) 
        #------------------------------------------
        
        vm0 = axs2[0,0].imshow(velmap, origin='lower',cmap='RdYlBu_r',aspect='equal',extent=[-12.5,12.5,-12.5,12.5],vmin=-120,vmax=215,zorder=0) # vmin=numpy.nanmin(velmap),vmax=numpy.nanmax(velmap)
        divider0 = make_axes_locatable(axs2[0,0])
        cax0 = divider0.append_axes('right', size='5%', pad=0.05)
        cbar0=fig2.colorbar(vm0, cax=cax0, orientation='vertical')
        cbar0.ax.tick_params(labelsize=7)
        cbar0.ax.set_ylabel(r'km/ $s^{-1}$',fontsize=12,**csfont, rotation=270,labelpad=8.8)

        'Show kin. axis from radon tr.:'
        extent_xmin = (-x_dim/2)*pixelres
        extent_xmax = (x_dim/2)*pixelres
        if PA>90:
            ymin = -extent_xmin*numpy.tan(numpy.pi-(PA*numpy.pi/180))
            ymax = -extent_xmax*numpy.tan(numpy.pi-(PA*numpy.pi/180))
        else:
            ymin = extent_xmin*numpy.tan(PA*numpy.pi/180)
            ymax = extent_xmax*numpy.tan(PA*numpy.pi/180)
        axs2[0,0].plot([extent_xmin,extent_xmax], [ymin,ymax], color='slategrey', linestyle='-', linewidth=3.9,alpha=1)
        #_________________________________


        axs2[0,0].set_xlabel('arcsec',fontsize=12,**csfont)
        axs2[0,0].set_ylabel('arcsec',fontsize=12,**csfont)

        axs2[0,0].scatter(0,0,s=90,color='black',marker='o',zorder=20)
        axs2[0,0].axvline(x=0,color='black',zorder=20,linewidth=1.2)
        axs2[0,0].axhline(y=0,color='black',zorder=20,linewidth=1.2)

        height, width = pixelres, pixelres
        
        axs2[0,0].xaxis.set_major_locator(MultipleLocator(5))
        axs2[0,0].xaxis.set_minor_locator(MultipleLocator(1))
        axs2[0,0].yaxis.set_major_locator(MultipleLocator(5))
        axs2[0,0].yaxis.set_minor_locator(MultipleLocator(1))
        axs2[0,0].tick_params(which='both',direction='in',width=0.5)

        x0    = -10
        xf    = x0 + r_app*pixelres
        y_arr = numpy.zeros(20) -10
        x_arr = numpy.linspace(x0,xf,num=20,endpoint=True)
        axs2[0,0].plot(x_arr,y_arr,color='forestgreen',linewidth = 5)
        
        #------------------------------------------
        vm1 = axs2[1,0].imshow(masked_map1, origin='lower',interpolation='none',cmap='BrBG',extent=[theta_min,theta_max,2*rho_min,2*rho_max],vmin=-2300,vmax=3100,zorder=0)   
        divider1 = make_axes_locatable(axs2[1,0])
        cax1 = divider1.append_axes('right', size='11%', pad=0.05)
        cbar1=fig2.colorbar(vm1, cax=cax1, orientation='vertical',shrink=0.25,aspect=10)
        cbar1.ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,labelsize=7)
        cbar1.ax.set_ylabel(r'km $s^{-1}$',fontsize=12,**csfont, rotation=270,labelpad=8.8) # OR cbar1.ax.set_title(r'km $s^{-1}$',fontsize=12,**csfont, rotation=0)
        axs2[1,0].set_ylabel(r'$/rho / $(arcsec)',fontsize=16,**csfont,rotation='90',labelpad=10) # Could also have x-axis title (none added since axes are shared): axs2[1,0].set_xlabel(r'$/theta /  $(degrees)',fontsize=12,**csfont)
        
        axs2[1,0].xaxis.set_major_locator(MultipleLocator(50))
        axs2[1,0].xaxis.set_minor_locator(MultipleLocator(25))
        axs2[1,0].yaxis.set_major_locator(MultipleLocator(20))
        axs2[1,0].yaxis.set_minor_locator(MultipleLocator(10))
        axs2[1,0].tick_params(which='both',direction='in',width=0.5)

        axs2[1,0].text(15,49,'R',fontsize=15,color='navy',**csfont,fontweight='bold')
        
        a1 = axs2[1,0].get_yticks().tolist()
        for i in range(len(a1)):
            a1[i]=float(a1[i])
            a1[i]=(a1[i]/2)*pixelres # Rho Labels in arcseconds --- remove '*pixelres' to convert to pixels 
        
        axs2[1,0].set_yticklabels(a1)
        
        #-------------------------------------------
        vm2 = axs2[2,0].imshow(masked_map2, origin='lower',cmap='seismic',aspect='equal',extent=[theta_min,theta_max,2*rho_min,2*rho_max],vmin=0,vmax=3600,zorder=0) 
        divider2 = make_axes_locatable(axs2[2,0])
        cax2 = divider2.append_axes('right', size='11%', pad=0.05)
        cbar2 = fig2.colorbar(vm2, cax=cax2, orientation='vertical')
        cbar2.ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,labelsize=7)
        cbar2.ax.set_ylabel(r'km $s^{-1}$',fontsize=12,**csfont, rotation=270,labelpad=15.8)    # cbar2.ax.set_title(r'km/ $s^{-1}$',fontsize=12,**csfont, rotation=0)
        axs2[2,0].set_xlabel(r'$/theta /  $(degrees)',fontsize=12,**csfont)
        axs2[2,0].set_ylabel(r'$/rho / $(arcsec)',fontsize=16,**csfont,rotation='90',labelpad=10)

        axs2[2,0].xaxis.set_major_locator(MultipleLocator(50))
        axs2[2,0].xaxis.set_minor_locator(MultipleLocator(25))
        axs2[2,0].yaxis.set_major_locator(MultipleLocator(20))
        axs2[2,0].yaxis.set_minor_locator(MultipleLocator(10))
        axs2[2,0].tick_params(which='both',direction='in',width=0.5)

        axs2[2,0].text(15,49,r'$R_{A}$',fontsize=15,color='navy',**csfont,fontweight='bold')

        a2 = axs2[2,0].get_yticks().tolist()
        for i in range(len(a2)):
            a2[i]=float(a2[i])
            a2[i]=(a2[i]/2)*pixelres # Rho Labels in arcseconds --- remove '*pixelres' to convert to pixels 
        
        
        axs2[2,0].set_yticklabels(a2)
        
        #-------------------------------------------
        vm3 = axs2[0,1].imshow(masked_map3, origin='lower',cmap='PuOr',aspect='equal',extent=[theta_min,theta_max,2*rho_min,2*rho_max],vmin=0,vmax=1,zorder=0) # vmin=-70,vmax=170  ||| vmin=numpy.nanmin(velrot_stellar),vmax=numpy.nanmax(velrot_stellar)
        divider3 = make_axes_locatable(axs2[0,1])
        cax3 = divider3.append_axes('right', size='11%', pad=0.05)
        cbar3=fig2.colorbar(vm3, cax=cax3, orientation='vertical')
        cbar3.ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,labelsize=7)
        #cbar3.ax.set_title(r'km $s^{-1}$',fontsize=12,**csfont, rotation=0)
        
        axs2[0,1].set_ylabel(r'$/rho / $(arcsec)',fontsize=16,**csfont,rotation='90',labelpad=10)

        axs2[0,1].xaxis.set_major_locator(MultipleLocator(50))
        axs2[0,1].xaxis.set_minor_locator(MultipleLocator(25))
        axs2[0,1].yaxis.set_major_locator(MultipleLocator(20))
        axs2[0,1].yaxis.set_minor_locator(MultipleLocator(10))
        axs2[0,1].tick_params(which='both',direction='in',width=0.5)

        axs2[0,1].text(15,49,r'$R_{A}/ $(rescaled)',fontsize=15,color='navy',**csfont,fontweight='bold')

        a3 = axs2[0,1].get_yticks().tolist()
        for i in range(len(a3)):
            a3[i]=float(a3[i])
            a3[i]=(a3[i]/2)*pixelres # Rho Labels in arcseconds --- remove '*pixelres' to convert to pixels 
        
        axs2[0,1].set_yticklabels(a3)

        #-------------------------------------------
        vm4 = axs2[1,1].imshow(masked_map4, origin='lower',cmap='PRGn',aspect='equal',extent=[theta_min,theta_max,2*rho_min,2*rho_max],vmin=80,vmax=810,zorder=0) # vmin=-70,vmax=170  ||| vmin=numpy.nanmin(velrot_stellar),vmax=numpy.nanmax(velrot_stellar)
        divider4 = make_axes_locatable(axs2[1,1])
        cax4 = divider4.append_axes('right', size='11%', pad=0.05)
        cbar4=fig2.colorbar(vm4, cax=cax4, orientation='vertical')
        cbar4.ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,labelsize=7)
        #cbar4.ax.set_title(r'km/ $s^{-1}$',fontsize=12,**csfont, rotation=0)
        cbar4.ax.set_ylabel(r'km $s^{-1}$',fontsize=12,**csfont, rotation=270,labelpad=15.8)
        axs2[1,1].set_ylabel(r'$/rho / $(arcsec)',fontsize=16,**csfont,rotation='90',labelpad=10)

        axs2[1,1].xaxis.set_major_locator(MultipleLocator(50))
        axs2[1,1].xaxis.set_minor_locator(MultipleLocator(25))
        axs2[1,1].yaxis.set_major_locator(MultipleLocator(20))
        axs2[1,1].yaxis.set_minor_locator(MultipleLocator(10))
        axs2[1,1].tick_params(which='both',direction='in',width=0.5)

        axs2[1,1].text(15,49,r'$R_{AB}$',fontsize=15,color='navy',**csfont,fontweight='bold')

        a4 = axs2[1,1].get_yticks().tolist()
        for i in range(len(a4)):
            a4[i]=float(a4[i])
            a4[i]=(a4[i]/2)*pixelres # Rho Labels in arcseconds --- remove '*pixelres' to convert to pixels 
        
        axs2[1,1].set_yticklabels(a4)

        #-------------------------------------------
        # Replace <masked_map5> with <mask_radon_AB> if you want to view the masked regions in R_AB (useful for debugging)
        vm5 = axs2[2,1].imshow(masked_map5, origin='lower',cmap='RdGy',aspect='equal',extent=[theta_min,theta_max,2*rho_min,2*rho_max],vmin=0,vmax=1,zorder=0) # vmin=-70,vmax=170  ||| vmin=numpy.nanmin(velrot_stellar),vmax=numpy.nanmax(velrot_stellar)
        
        divider5 = make_axes_locatable(axs2[2,1])
        cax5 = divider5.append_axes('right', size='11%', pad=0.05)
        cbar5=fig2.colorbar(vm5, cax=cax5, orientation='vertical')
        cbar5.ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,labelsize=7)
        #cbar5.ax.set_title(r'km $s^{-1}$',fontsize=12,**csfont, rotation=0)
        axs2[2,1].set_xlabel(r'$/theta /  $(degrees)',fontsize=12,**csfont)
        axs2[2,1].set_ylabel(r'$/rho / $(arcsec)',fontsize=16,**csfont,rotation='90',labelpad=10)

        axs2[2,1].xaxis.set_major_locator(MultipleLocator(50))
        axs2[2,1].xaxis.set_minor_locator(MultipleLocator(25))
        axs2[2,1].yaxis.set_major_locator(MultipleLocator(20))
        axs2[2,1].yaxis.set_minor_locator(MultipleLocator(10))
        axs2[2,1].tick_params(which='both',direction='in',width=0.5)

        axs2[2,1].text(15,49,r'$R_{AB}/ $(rescaled)',fontsize=15,color='navy',**csfont,fontweight='bold')

        a5 = axs2[2,1].get_yticks().tolist()
        for i in range(len(a5)):
            a5[i]=float(a5[i])
            a5[i]=(a5[i]/2)*pixelres # Rho Labels in arcseconds --- remove '*pixelres' to convert to pixels 
        
        axs2[2,1].set_yticklabels(a5)

        
        #_________________________________________________________________________________________________________________________________________________
        #                                                                  RADON PROFILE
        #_________________________________________________________________________________________________________________________________________________

        fig3, axs3 = plt.subplots(1,1,figsize=(40,40))
        plt.subplots_adjust(left=0.193, bottom=0.507, right=0.876, top=0.852, wspace=0, hspace=0.183) # bottom = 0.207

        if component == 'stars':
            axs3.scatter(rho_arr_rp*pixelres,theta_hat_rp,s=120,color='lightseagreen',marker='o',edgecolor='navy',zorder=0)
            axs3.plot(rho_arr_rp*pixelres,theta_hat_rp,color='lightseagreen',linewidth=7,zorder=-10,linestyle='-')
        elif component == 'gas':
            axs3.scatter(rho_arr_rp*pixelres,theta_hat_rp,s=120,color='lightcoral',marker='o',edgecolor='navy',zorder=0)
            axs3.plot(rho_arr_rp*pixelres,theta_hat_rp,color='lightcoral',linewidth=7,zorder=-10,linestyle='-')
        
        axs3.plot(rho_arr_rp*pixelres,theta_hat_rp,color='black',linewidth=10,zorder=-20,linestyle='-')
        #____________
        'Shows theta_flipped'
        theta_hat_flipped = numpy.flip(theta_hat_rp)
        rho_flipped = numpy.flip(-1*(rho_arr_rp)) # rho_arr_rp / rho_flipped
        #axs3.scatter(rho_flipped*pixelres,theta_hat_flipped,s=120,color='crimson',marker='o',edgecolor='black')
        rho1 = numpy.around(rho_arr_rp,decimals=3)
        rho2 = numpy.around(rho_flipped,decimals=3)
        rho_com = numpy.intersect1d(rho1, rho2)
        if len(rho_com)!=0:
            min_rho_com = numpy.nanmin(rho_com)*pixelres
            max_rho_com = numpy.nanmax(rho_com)*pixelres
        else:
            min_rho_com = 0
            max_rho_com = 0
        #____________

        axs3.set_xlabel(r'$/rho / $(arcsec)',fontsize=28,**csfont)
        axs3.set_ylabel(r'$/theta_{min} /  $(degrees)',fontsize=28,**csfont,rotation='90',labelpad=10,zorder=0) # r'$/rho / $(arcsec)'
        axs3.axhline(y=mean_theta,linestyle='--',linewidth=4,zorder=-10,color = 'mediumspringgreen')
        axs3.axhspan(mean_theta-stdev_theta, mean_theta+stdev_theta, alpha=0.3, color='aquamarine',zorder=-200)
        axs3.axhline(y=mean_theta-stdev_theta,linestyle='-',linewidth=1,zorder=-10,color = 'grey')
        axs3.axhline(y=mean_theta+stdev_theta,linestyle='-',linewidth=1,zorder=-10,color = 'grey')

        axs3.axvline(x=0,linestyle='--',linewidth=4,zorder=-10,color = 'seagreen')
        axs3.axvline(x=min_rho_com,linestyle='-',linewidth=3,zorder=-10,color = 'black')
        axs3.axvline(x=max_rho_com,linestyle='-',linewidth=3,zorder=-10,color = 'black')
        axs3.tick_params(which='both',direction='in',width=1.7,labelsize=18)
        
        
        if component == 'stars':
            print('SAVING - STARS')
            plt.savefig('/Users/23211651/Desktop/Barred/Stars/G_'+str(gal_ind))
        elif component == 'gas':
            plt.savefig('/Users/23211651/Desktop/Barred/Gas/G'+str(gal_ind))
            print('SAVING - GAS')


        print('FINAL TIME - BEFORE RETURN')
        print("--- %s seconds ---" % (time.time() - start_time))


    #__________________________________________________________________________________
    #__________________________________________________________________________________
    #    theta_hat_rp is measured from EAST = 0 counterclockwise
    #  Correct it to be measured from NORTH = 0 counterclockwise to REDSHIFTES size
    #__________________________________________________________________________________
    #__________________________________________________________________________________
    theta_hat_rp_N   = numpy.zeros(len(theta_hat_rp))
    theta_hat_rp_360 = numpy.zeros(len(theta_hat_rp))

    #rot_PA            = rotate(velmap, 90 + theta_hat_rp_N[i] , reshape=False,order=0)
    #mean_vm_left        = numpy.nanmean(rot_PA_i[:,0:int(y_dim/2)]) 
    #mean_vm_right       = numpy.nanmean(rot_PA_i[:,int(y_dim/2):y_dim]) 
    for i in range(len(theta_hat_rp)):
        if theta_hat_rp[i] >= 90:
            theta_hat_rp_N[i] = theta_hat_rp[i] - 90
        elif theta_hat_rp[i]!=-1:
            theta_hat_rp_N[i] = theta_hat_rp[i] + 90
        
        theta_hat_rp_360[i]              = -1

        if mean_vm_left > mean_vm_right:
            theta_hat_rp_360[i] = theta_hat_rp_N[i] + 180
        else:
            theta_hat_rp_360[i] = theta_hat_rp_N[i]
    #__________________________________________________________________________________
    #__________________________________________________________________________________
    

    return rho_arr,theta_arr, rd ,rd_A, rd_A_scaled , rd_AB, rd_AB_scaled, rho_arr_rp, theta_hat_rp ,Nij, PA_N, PA_360
    
#_______________________________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________
#_______________________________________________________________________________________________________________________________________________________

x_grid = numpy.linspace(-1.5,1.5,num=3, endpoint=True) # num = 9 # / ! in arcsec !
y_grid = numpy.linspace(-1.5,1.5,num=3, endpoint=True) # num = 9 # / ! in arcsec !

def recentering(velmap,pixelres,x_grid,y_grid,apperture,plot='no'):
    #_______________________________________________________________________________________
    #  Given a velocity map and a grid defined by the x_grid & y_grid arrays (x/y coordinates of the grid points), the function recentres the vel. map on each point 
    #  of the grid, then computes Radon transform maps, Radon profiles and calculates the Asymmetry parameter (see )
    #_______________________________________________________________________________________
    #            velmap          : Velocity map to be recentred around the grid points, where the radon transform maps/profiles are computed & asymmetries calculated
    #           pixelres         : Pixel resolution of the velocity map
    #       x_grid / y_grid      : coordinates of the grid where the vel. map is to be recentered in order to look for the position of the kin. centre (!in arcsec!)
    #         apperture          : Radon apperture to be used for running the Radon function   
    #   plot(kw, default = 'no') : if =='yes', it will produce a plot of the grid points and the asymmetry calculated at each point AND a plot of the vel. map centrea at EACH point on the grid
    #_______________________________________________________________________________________
    
    x_dim        = len(velmap[:,0])   # = 50 for SAMI
    y_dim        = len(velmap[0,:])   # = 50 for SAMI
    x_dim_arcsec = x_dim*pixelres     # = 25'' for SAMI
    y_dim_arcsec = y_dim*pixelres     # = 25'' for SAMI
    
    'Initial test - imputed grid needs to be at least 3x3:'
    if len(x_grid) < 3 or len(y_grid) < 3:
        print ('Error - grid imputed into recentering function needs to be at least 3x3') 
        return -1, -1, -1, -1

    
    x0,x1 = x_grid[0], x_grid[len(x_grid)-1]
    y0,y1 = y_grid[0], y_grid[len(y_grid)-1]
    step_x = x_grid[1] - x_grid[0]
    step_y = y_grid[1] - y_grid[0]

    grid_w = y_grid[len(y_grid)-1] - y_grid[0]
    grid_h = x_grid[len(x_grid)-1] - x_grid[0]
    
    x_grid = x_grid/pixelres    # In pixels
    y_grid = y_grid/pixelres    # In pixels
    
    
    xdim   = (len(velmap[0,:])/2)*pixelres
    ydim   = (len(velmap[:,0])/2)*pixelres
    #___________________________________
    rho_arr0, theta_arr0, rd0 ,rd_A0, rd_A_scaled0 ,rd_AB0, rd_AB_scaled0, rho_arr_rp0, theta_hat_rp0, N00, PA_N0, PA_360o = radon(velmap, 1, 1, 0.5, apperture,100000, plot='no') # 6
    Asym_arr = numpy.zeros((len(x_grid), len(y_grid))) + 9999
    
    plt.savefig('/Users/23211651/Desktop/Barred/P2/Gal_recdntered_') #  + str(plate) + '_' + str(ifu) ) # PA_Determination_Plots
    plt.close('all')
    if plot=='yes':
        fig_rec, axs_rec   = plt.subplots(ncols=len(x_grid), nrows=len(y_grid),figsize=(97,97)) #plt.subplots(1,5,figsize=(97,97))
        plt.subplots_adjust(left=0.136, bottom=0.06, right=0.862, top=0.983, wspace=0, hspace=0) 
        fig_asym, axs_asym = plt.subplots(ncols=2, nrows=1,figsize=(97,97)) #plt.subplots(1,5,figsize=(97,97))

    centre_grid = (len(x_grid)-1)/2 # Coordinate of the centre of the grid (same in x/y if grid is symmetric about image centre - as it should be)
    for i in range(len(y_grid)):
        # Reflect the grid w.r.t the centre. This is done because, if the map is fooset by e.g. x = y = 2'', then the map will be centred on xc = yc = -2''
        if i > centre_grid or i < centre_grid:
            y_coord =  len(y_grid)-1-i
        elif i == centre_grid:
            y_coord = i
        #___________________________
        for j in range(len(x_grid)):
            shift = numpy.array([y_grid[i],x_grid[j]])
            if x_grid[j]==0 and y_grid[i]==0:
                
                velmap_shift = velmap
            else:
                velmap_shift = shift_image(velmap,pixelres,shift)
            
            rho_arr_i, theta_arr_i, rd_i ,rd_A_i, rd_A_scaled_i ,rd_AB_i, rd_AB_scaled_i, rho_arr_rp_i, theta_hat_rp_i, Nij_i, PA_Ni, PA_360i = radon(velmap_shift, 1, 1, 0.5, 6,100000, plot='no') # r_app = 3/6 
            Asym, wi, Ni, rho_com_mini, rho_com_maxi = asymmetry (theta_hat_rp_i, rho_arr_rp_i, N00)
            
            # Reflect the grid w.r.t the centre. This is done because, if the map is fooset by e.g. x = y = 2'', then the map will be centred on xc = yc = -2''
            if j > centre_grid or j < centre_grid:
                x_coord =  len(x_grid)-1-j
            elif j == centre_grid:
                x_coord = j
            

            Asym_arr[y_coord,x_coord] = Asym   # [len(x_grid)-1-i,len(y_grid)-1-j]     #   x_coord  ,  y_coord

            #_____________________________________________________________________________
            #                                  PLOTS                                      |
            #_____________________________________________________________________________|
            if plot=='yes':
                app = x_grid[len(x_grid)-1]*pixelres
                axs_rec[i,j].imshow(velmap_shift,origin='lower',cmap='PRGn',aspect='equal',extent=[-xdim,xdim,-ydim,ydim],vmin=numpy.nanmin(velmap_shift),vmax=numpy.nanmax(velmap_shift),zorder=0)
                rec_circle    = plt.Circle((0 + shift[1]*pixelres, 0 + shift[0]*pixelres), app,edgecolor='black',fill=False,linestyle='--',linewidth=1.8,zorder=10)
                rec_rectangle = plt.Rectangle((0 - grid_h/2 + shift[1]*pixelres    ,    0 - grid_w/2  + shift[0]*pixelres ), grid_h, grid_w, fc='none',ec='crimson',linestyle='--')
                axs_rec[i,j].add_artist(rec_circle)
                axs_rec[i,j].add_artist(rec_rectangle)
                axs_rec[i,j].scatter(0,0,s=30,color='orangered',edgecolor='black',marker='o',zorder=20)
                axs_rec[i,j].axvline(x=0 + shift[1]*pixelres,color='black',zorder=20,linewidth=1.2)
                axs_rec[i,j].axhline(y=0 + shift[0]*pixelres,color='black',zorder=20,linewidth=1.2)
                axs_rec[i,j].text(-0.85*ydim,0.85*xdim,   'x='+ str(x_grid[j]*pixelres) + '|' + 'y=' + str(y_grid[i]*pixelres)  ,fontsize=9,color='mediumblue')
                axs_rec[i,j].text(-0.85*ydim,0.65*xdim,   str(numpy.around(Asym_arr[y_coord,x_coord], decimals=2))  ,fontsize=9,color='red')
    
            
    if plot=='yes':
        asym_map = axs_asym[0].imshow(Asym_arr,origin='lower',cmap='PRGn',aspect='equal',extent=[x0-step_x/2,  x1+step_x/2,  y0-step_y/2,  y1+step_y/2],vmin=numpy.nanmin(Asym_arr),vmax=numpy.nanmax(Asym_arr),zorder=0)
        axs_asym[0].set_xlabel(r'$x/ (arcsec)$',fontsize=20)
        axs_asym[0].set_ylabel(r'$y/ (arcsec)$',fontsize=20)
        divider_asym = make_axes_locatable(axs_asym[0])
        cax_asym = divider_asym.append_axes('right', size='11%', pad=0.05)
        cbar_asym=fig_asym.colorbar(asym_map, cax=cax_asym, orientation='vertical')
        cbar_asym.ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,labelsize=7)
        cbar_asym.ax.set_title(r'Asymmetry',fontsize=12, rotation=0)
        
        #______________________________
        szx = len(Asym_arr[0,:])
        szy = len(Asym_arr[:,0])
        jump_x = (x1 - x0) / (2.0 * szx)
        jump_y = (y1 - y0) / (2.0 * szy)
        
        x_positions = numpy.linspace(start=x0, stop=x1, num=szx, endpoint=True)
        y_positions = numpy.linspace(start=y0, stop=y1, num=szy, endpoint=True)
        
        extra_x = step_x/len(x_grid) - 1
        extra_y = step_y/len(y_grid) - 1
        for y_index, y in enumerate(y_positions):
            for x_index, x in enumerate(x_positions):
                label = numpy.around(Asym_arr[y_index, x_index],decimals=2)
                text_x = x #+ jump_x 
                text_y = y #+ jump_y 
                axs_asym[0].text(text_x, text_y, str(label), ha='center', va='center',fontsize=14,color='crimson')

    min_asym = numpy.nanmin(Asym_arr)
    
    x_centre0, y_centre0 = 0,0                    # Coordinates of the minima in the Asym. array
    
    minpos = numpy.argwhere(Asym_arr == min_asym)
    x_centre0 = x_grid[minpos[0,1]]*pixelres
    y_centre0 = y_grid[minpos[0,0]]*pixelres
    
    #__________________________________________________________
    # Extra - take weighted positional average of positions with asymmetries below average/mean 
    #_______________________________________________________________________________________________
    # asym_belowav | asym_belowmean : Asymmetry values below average/mean
    # x_belowav    | x_belowmean    : x-coordinates on the grid of asymmetry values below average/mean
    # y_belowav    | y_belowmean    : y-coordinates on the grid of asymmetry values below aberage/mean
    #_______________________________________________________________________________________________
    asym_belowav , asym_belowmean, x_belowav  , x_belowmean, y_belowav , y_belowmean = [],[],[],[],[],[]
    average_asym = numpy.average(Asym_arr)
    mean_asym    = numpy.nanmean(Asym_arr)
    
    arg_asym_belowmean   = numpy.argwhere(Asym_arr < mean_asym)
    asym_belowmean       = Asym_arr[arg_asym_belowmean[:,0],arg_asym_belowmean[:,1]]
    x_belowmean          = x_grid[arg_asym_belowmean[:,1]]
    y_belowmean          = x_grid[arg_asym_belowmean[:,0]]

    arg_asym_belowav   = numpy.argwhere(Asym_arr < average_asym)
    asym_belowav       = Asym_arr[arg_asym_belowav[:,0],arg_asym_belowav[:,1]]
    x_belowav          = x_grid[arg_asym_belowav[:,1]]
    y_belowav          = x_grid[arg_asym_belowav[:,0]]

    weighted_x_belowmean , weighted_y_belowmean  = 0,0
    weighted_x_belowav , weighted_y_belowav  = 0,0
    
    'Using the mean:'
    '__________________________________________________________________'
    if len(asym_belowmean) >= 1:
        weight_belowmean = numpy.nanmin(asym_belowmean)/asym_belowmean
    else:
        weight_belowmean = 0
    
    weighted_x_belowmean , weighted_y_belowmean  = 0*pixelres,0*pixelres

    for i in range(len(asym_belowmean)):
        weighted_x_belowmean = weighted_x_belowmean + (x_belowmean[i]*weight_belowmean[i])
        weighted_y_belowmean = weighted_y_belowmean + (y_belowmean[i]*weight_belowmean[i])
    
    weighted_x_belowmean = (weighted_x_belowmean/sum(weight_belowmean))*pixelres
    weighted_y_belowmean = (weighted_y_belowmean/sum(weight_belowmean))*pixelres

    'Using the average:'
    '__________________________________________________________________'
    if len(asym_belowav) >= 1:
        weight_belowav = numpy.nanmin(asym_belowav)/asym_belowav
    else:
        weight_belowav = 0
    
    for i in range(len(asym_belowav)):
        
        weighted_x_belowav = weighted_x_belowav + (x_belowav[i]*weight_belowav[i])
        weighted_y_belowav = weighted_y_belowav + (y_belowav[i]*weight_belowav[i])
    
    weighted_x_belowav = (weighted_x_belowav/sum(weight_belowav))*pixelres
    weighted_y_belowav = (weighted_y_belowav/sum(weight_belowav))*pixelres

    #_______________________________________________________________________________________________
    #                    SECONDARY PLOT - SHOW CENTRE OFFSET
    #_______________________________________________________________________________________________
    axs_asym[1].imshow(velmap, origin='lower',cmap='RdYlBu_r',aspect='equal',extent=[-x_dim_arcsec/2,x_dim_arcsec/2,-y_dim_arcsec/2,y_dim_arcsec/2],vmin=numpy.nanmin(velmap),vmax=numpy.nanmax(velmap),zorder=0) # vmin=-70,vmax=170  ||| vmin=numpy.nanmin(velrot_stellar),vmax=numpy.nanmax(velrot_stellar)
    axs_asym[1].axhline(y=0,color='black',linewidth=1.2,zorder=20)
    axs_asym[1].axvline(x=0,color='black',linewidth=1.2,zorder=20)
    axs_asym[1].scatter(x_centre0,y_centre0,color='aquamarine',edgecolor='black',s=50,marker='o',zorder=50, label = 'min. asymmetry')
    axs_asym[1].scatter(weighted_x_belowmean,  weighted_y_belowmean,color='crimson',edgecolor='black',s=50,marker='o',zorder=50, label = 'weighted - above mean')
    axs_asym[1].scatter(weighted_x_belowav,  weighted_y_belowav,color='darkorchid',edgecolor='black',s=2150,marker='o',zorder=50, label = 'weighted - above mean')

    axs_asym[1].tick_params(which='both',direction='in',width=0.5,labelsize=6)

    height,width = step_x, step_y
    
    sq0 = mpatches.Rectangle((x_centre0 - step_x/2 - 0.02 , y_centre0 - step_y/2 - 0.02),height+0.02,width+0.02,fill=False,color='crimson',linestyle='-',linewidth=1.83,zorder=30)
    axs_asym[0].add_artist(sq0)

    'Re-centering using weighted below-average positions have been discontinued'

    for i in range(len(x_belowmean)):
        sq2 = mpatches.Rectangle((x_belowmean[i]*pixelres - step_x/2 - 0.02 , y_belowmean[i]*pixelres - step_y/2 - 0.02),height+0.02,width+0.02,fill=False,color='orange',linestyle='-',linewidth=8.5,zorder=10)
        axs_asym[0].add_artist(sq2)


    plt.savefig('/Users/23211651/Desktop/Barred/P3/G_') #  + str(plate) + '_' + str(ifu) ) # PA_Determination_Plots
    plt.close('all')
    return Asym_arr, x_centre0, y_centre0, weighted_x_belowav, weighted_y_belowav, weighted_x_belowmean, weighted_y_belowmean







#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
# Code to test the running of the radon function

'''
directory = '/Users/23211651/Desktop/PhD/SAMI_Galaxies/sami_ALL/dr3/ifs'
count_g = 0
for entry in os.scandir(directory):              # FOR READING ALL SAMI GALAXIES IN THE DIRECTORY

    gal_index =   entry.name                     # INDEX OF THE GALAXY 
    newdir    =   entry.path                  
    
    if gal_index == '.DS_Store':
        continue
    
    for entry2 in os.scandir(newdir):
        dirpath=entry2.path
        break
    count_g=count_g+1
    full_filename_prefix=dirpath[0:60+2*int(len(gal_index))] # 38+int(len(gal_index)) # 39 (supercomp. file) ---> 60 (macbook file)

    if gal_index==str(376478): # 15218 # 15165 # 14812 # 16487 # 9067 #                         22932
        # STELLAR ROT #
        fname_2mdefault         = full_filename_prefix + '_A_stellar-velocity_default_two-moment.fits'
        hdul_stellar_rot        = fits.open(fname_2mdefault)           
        data_stellar_rotation   = hdul_stellar_rot[0].data
        error_stellar_rotation  = hdul_stellar_rot[1].data
        S_N_stellar             = hdul_stellar_rot[4].data
        # STELLAR DISP #
        fname_stellar_disp      = full_filename_prefix + '_A_stellar-velocity-dispersion_default_two-moment.fits'   #    X
        hdul_stellar_disp       = fits.open(fname_stellar_disp)
        data_stellar_disp       = hdul_stellar_disp[0].data
        error_stellar_disp      = hdul_stellar_disp[1].data
        # GAS ROT #
        hdul_gas_rot = fits.open(full_filename_prefix + '_A_gas-velocity_default_1-comp.fits')
        data_velrot_gas = hdul_gas_rot[0].data
        error_velrot_gas = hdul_gas_rot[1].data
        # GAS DISP # 
        fname_gas_disp   = full_filename_prefix + '_A_gas-vdisp_default_1-comp.fits'
        hdul_gas_disp    = fits.open(fname_gas_disp)
        data_gas_disp    = hdul_gas_disp[0].data
        error_gas_disp   = hdul_gas_disp[1].data
        # HALPHA # 
        fname_Halpha      = full_filename_prefix + '_A_Halpha_default_1-comp.fits'                            #      X
        hdul_Halpha       = fits.open(fname_Halpha)
        Halpha_flux       = hdul_Halpha[0].data
        Halpha_flux_error = hdul_Halpha[1].data
        #__________________________________________
        hdul_stellar_rot.close()
        hdul_stellar_disp.close()
        hdul_gas_rot.close()
        hdul_gas_disp.close()
        hdul_Halpha.close()
        #____________________________________|
        #____________________________________|
        cubesizex = shape(Halpha_flux)[1]                 # image is 50 x 50 pixels for SAMI
        cubesizey = shape(Halpha_flux)[2]                 # cubesizex,cubesizey = 50,50 # FOR SAMI
        
        Halpha_fractional_error = numpy.zeros([cubesizex,cubesizey])
        for i in range(cubesizex):
            for j in range(cubesizey):
                if Halpha_flux[0,i,j] != 0:
                    Halpha_fractional_error[i,j] = Halpha_flux_error[0,i,j] / Halpha_flux[0,i,j]
                else:
                    Halpha_fractional_error[i,j] = 0
                if (numpy.isnan(Halpha_flux[0,i,j]) == True or numpy.isnan(Halpha_flux_error[0,i,j]) == True) and numpy.isnan(data_velrot_gas[0,i,j]) == False:
                    Halpha_fractional_error[i,j] = 0

        #_________________________________________________________________________________________________________________________
        #                                                  EXCLUSION OF PIXELS (QUALITY CUTS) 
        #_________________________________________________________________________________________________________________________
        for i in range(cubesizex):
            for j in range(cubesizey):
                #_________________________________________________________________________________________________________________
                #                                                     STARS 
                #_________________________________________________________________________________________________________________
                if error_stellar_rotation[i,j]>50:# or error_stellar_rotation[i,j]>0.7*numpy.absolute(data_stellar_rotation[i,j]):                   # the value is based on SAMI DR3 advice : |||| 1*numpy.absolute(data_stellar_rotation[i,j]):
                    data_stellar_rotation[i,j]=numpy.nan
                    error_stellar_rotation[i,j]=numpy.nan
                if S_N_stellar[i,j]<5:
                    data_stellar_rotation[i,j]=numpy.nan
                    error_stellar_rotation[i,j]=numpy.nan
                if data_stellar_disp[i,j]<35:
                    data_stellar_rotation[i,j]=numpy.nan
                    error_stellar_rotation[i,j]=numpy.nan              
                if error_stellar_disp[i,j]>0.1*data_stellar_disp[i,j]+30:
                    data_stellar_rotation[i,j]=numpy.nan
                    error_stellar_rotation[i,j]=numpy.nan    
                
                #_________________________________________________________________________________________________________________
                #                                                      GAS 
                #_________________________________________________________________________________________________________________
                if error_velrot_gas[0,i,j]>=50:# 20: |||| 1*numpy.absolute(data_velrot_gas[0,i,j]):
                    data_velrot_gas[0,i,j]=numpy.nan
                    error_velrot_gas[0,i,j]=numpy.nan
                if Halpha_fractional_error[i,j]>=0.9:
                    data_velrot_gas[0,i,j]=numpy.nan
                    error_velrot_gas[0,i,j]=numpy.nan
                if data_gas_disp[0,i,j]<15:
                    data_velrot_gas[0,i,j]=numpy.nan
                    error_velrot_gas[0,i,j]=numpy.nan
                if error_gas_disp[0,i,j]>0.25*data_gas_disp[0,i,j]+15:
                    data_velrot_gas[0,i,j]=numpy.nan
                    error_velrot_gas[0,i,j]=numpy.nan

'''  
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________
#________________________________________________________________________________________________________________________________________________________________

