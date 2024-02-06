#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 22:45:22 2023

Code to plot the results for the demo MCMC

@author: Weigang Liang
"""


import matplotlib.pyplot as plt
import numpy as np
import scipy.io as io
import time
from scipy.interpolate import UnivariateSpline


start_time = time.time()



    
grav_col1 = 120
grav_col2 = 200
grav_row1 = 150
grav_row2 = 240
y_cen = 196
x_cen = 160
ys = 256e3
    


def plott(td,xs,th):

    
    from_matlab = {}
    

    from_matlab = io.loadmat('nne.mat')
    
        
    gravv0 = np.squeeze(from_matlab['grav2'])

    gravv = np.mean(gravv0[:,grav_col1:grav_col2]*1e5,axis=1)
    gravv = gravv - min(gravv[grav_row1:grav_row2])
    
    std_p01 = np.std(gravv0[:,grav_col1:grav_col2]*1e5,axis=1)#900:950

    
    G = 6.67408e-11;
    drho = 400;
    

    topo0 = 1738000*np.ones((1002,2004));
    
    lat = np.transpose(np.tile(np.linspace(0,180-180/1002,1002),(2004,1)))*np.pi/180;
    lon = np.tile(np.linspace(0,360-360/2004,2004),(1002,1))*np.pi/180;
    xp = topo0*np.sin(lat)*np.cos(lon);
    yp = topo0*np.sin(lat)*np.sin(lon);
    zp = topo0*np.cos(lat);
    
    latc = lat[y_cen-1,x_cen-1];
    lonc = lon[y_cen-1,x_cen-1];
    z_trans = np.array(((np.cos(lonc),np.sin(lonc),0),
        (-np.sin(lonc), np.cos(lonc), 0),
        (0, 0, 1)))
    y_trans = np.array(((np.cos(latc), 0, -np.sin(latc)),
        (0, 1, 0),
        (np.sin(latc), 0, np.cos(latc))))
    trans = np.matmul(y_trans,z_trans);
            
    xr = np.reshape(xp,(1,2004*1002)); yr = np.reshape(yp,(1,2004*1002)); zr = np.reshape(zp,(1,2004*1002));
    cr = np.matmul(trans,np.squeeze(np.stack((xr,yr,zr))));
    xp0 = np.reshape(cr[0,:],(1002,2004));
    yp0 = np.reshape(cr[1,:],(1002,2004));
    zp0 = np.reshape(cr[2,:],(1002,2004));
    
    xp = np.reshape(xp0[:,x_cen-1],(1002,1))
    yp = np.reshape(yp0[:,x_cen-1],(1002,1))
    zp = np.reshape(zp0[:,x_cen-1],(1002,1))
    

    y = np.array((-ys/2,ys/2)); 
  
      
    def forward_grav(td,th,xs,ys,div):
        grav = np.zeros((1002,1));
        grav_f= np.zeros((1002,2004))
    
        
        x_s = np.flip((np.arange(div)+1)*xs/div/2)
        z_s = td+(np.arange(div+1))*th/div
    
        
        for n in np.arange(div):
            x = np.array((-x_s[n],x_s[n]));    
            z = np.array((1738e3-z_s[n],1738e3-z_s[n+1]))
        
            for i in np.arange(2):
                dxi = x[i]-xp;
                for j in np.arange(2):
                    dyj = y[j]-yp;
                    for k in np.arange(2):
                        mu = (-1)**i*(-1)**j*(-1)**k;
                        dzk = z[k]-zp;
                        R = (dxi**2 + dyj**2 + dzk**2)**0.5;
                        grav = grav + mu*(dzk*(np.arctan(dxi*dyj/(dzk*R)))
                            - dxi*np.log(R+dyj)
                            - dyj*np.log(R+dxi));
        
                    
        
        grav_f = G*drho*np.repeat(grav,2004,axis=1);
        return grav_f
          
    grav_p0 = forward_grav(td,th,xs,ys,100)
    grav_p01 = np.mean(grav_p0[:,grav_col1:grav_col2]*1e5,axis=1)
        
    cc = np.arange(-500,502)/500*2728.4732196427353
    
    xlims = np.arange(grav_row1-600,grav_row2-400)/500*2728.4732196427353
    cc2 = cc-xlims[0]

    leg = ['sdfds']
 
    leg[0] = 'Gravity Anomaly'
    filename = 'NNE_mcmc_demo.eps'
    error = 'mcmc_error_demo.npy'

    
    profs = np.load(error)
    mcmc_err = np.std(profs,axis=0)
    
    plt.figure(figsize=(20,10))
    ax = plt.axes([0, 0.05, 0.9, 0.9 ])

    plt.plot(cc2,gravv,'k',linewidth=3)
    plt.errorbar(cc2,gravv,yerr=std_p01,fmt='k',elinewidth=0.5)
    plt.plot(cc2,-grav_p01,'magenta',linewidth=3)
    plt.errorbar(cc2,-grav_p01,yerr=mcmc_err,fmt='magenta',elinewidth=0.5)
    
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    max_y = max(-grav_p01)+max(mcmc_err)




    leg.extend(['Best-fit Model'])
    
    plt.legend(leg,fontsize=24)

    plt.xlim((0,xlims[-1]-xlims[0]))
    plt.ylim((-100,max(300,max_y)))
    plt.xlabel('Distance across strike (km)',fontsize=32)
    plt.ylabel('Gravity (mGal)',fontsize=32)        