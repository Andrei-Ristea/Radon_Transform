# Radon_Transform
Implementing the Radon Transform (line integral of a 2D function) to galaxy kinematics

This repository contans an all-encompassing python routine Radon9.py, which implements the Radon Transform (a line integral of a 2D function) to galaxy kinematic maps. For a full description of the application of Radon transforms to galaxy kinematics see Stark et al 2018 (https://academic.oup.com/mnras/article/480/2/2217/5061639?login=false), and references therein). 

The python script is heavily commented, with ample descriptions. The most useful function is:

--- radon(velmap,dx,dy,pixelres,r_app,gal_ind, plot='no' , plot_at_rho = 3.55,component='stars')
This function performs a radon transform to the 2D galactic velocity map and returns the following:
- rho_arr,theta_arr: The arrays of radii (rho_arr) and angles (theta_arr, from East = 0 counterclockwise) that define the directions along which the radon transforms are performed
- rd_A, rd_A_scaled , rd_AB, rd_AB_scaled, rho_arr_rp: Various values of the line integral (radon transforms; see function comments for a detailed desctiption of each transform and its uses) at different radii and angles.
- rho_arr_rp, theta_hat_rp: The values of radius and theta defining the direction along which the asymmetry of the velocity map was computed
- Nij: The asymmetry of the velocity map at different radii and angles
- PA_N, PA_360: The position angle of the main kinematic axis of the velocity map (PA_N is measured from N=0 deg, counterclockwise until the kinematic axis is reached; PA_360 is also measured from N=0 deg, until the receding side of the galactic disc, to identify the galaxy's rotational direction). 

  
