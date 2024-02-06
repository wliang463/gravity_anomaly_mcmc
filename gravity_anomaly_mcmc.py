#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Code to perform the Markov chain Monte Carlo method using the 
Metropolis-Hastings algorithm on GRAIL gravity data to constrain
the dimensions of the IBC bodies beneath the lunar surface

Units in this work are in SI units

Needs gravity files in .mat form

@author: Weigang Liang
"""

import numpy as np
import scipy.io as io
import copy
import pandas as pd
from math import log10, floor

################################################################################
#Defining global variables
################################################################################

G = 6.67408e-11
drho = 400;#density contrast between mantle and IBC in kg/m^3
mcmc_runs = 5#number of MCMC runs per anomaly
total_iterations = 10000 #number of iterations in each MCMC run
div_num = 100 #number of rectangular prisms the triangular prism is divided into

#replace with correct files/paths as needed
nw_grav = 'nnw.mat'
ne_grav = 'nne.mat'
s_grav = 'ns.mat'

#Information for the various gravity anomalies of the Moon. 
#The NW, NE, and S anomalies are in the first, second, and third indices, respectively.

grav_col1_0 = np.array((700,120,1872))
grav_col2_0 = np.array((850,200,1940))
grav_row1_0 = np.array((291,150,627))
grav_row2_0 = np.array((360,240,682))
y_cen_0 = np.array((333,196,641))
x_cen_0 = np.array((776,160,1906))
l_0 = np.array((702e3,256e3,336e3))

################################################################################
#Defining functions used in this work
################################################################################

#function to calculate gravity from a rectangular prism based on Nagy 1966

#td: top depth
#th: thickness
#w: width
#l: length
#xp, yp, zp: coordinates centered at the anomaly
#div: 

def grav_calc(td,th,w,xp,yp,zp,div):
        grav = np.zeros((1002,1));
        grav_f= np.zeros((1002,2004))
        
        x_s = np.flip((np.arange(div)+1)*w/div/2)
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

#defining the likelihood function as part of the MCMC algorithm

def likelihood(td,bd,w,grav):
    
    model_grav = grav_calc(td,bd-td,w,xp,yp,zp,div_num)
    model_prof = np.mean(model_grav[:,grav_col1:grav_col2]*1e5,axis=1)
     
    rms = (np.mean((grav[grav_row1:grav_row2]+model_prof[grav_row1:grav_row2])**2))**0.5
    
    llh = np.exp(np.float64(-rms**2/297))    
 
    return llh, rms


#captures the 16% and 84% percentile bounds for the posterior distribution

def r(x, sig=3):
    return 0 if x == 0 else round(x, sig - int(floor(log10(abs(x)))) - 1)

def lb(chain): return np.sort(chain)[int((len(chain) - 1) * 0.16)]

def rb(chain): return np.sort(chain)[int((len(chain) - 1) * 0.84)]


##################################################################################
#MCMC runs, loops through the three anomalies of interest
##################################################################################

for anomaly_ind in np.array([0,1,2]):
    
    grav_col1 = grav_col1_0[anomaly_ind];
    grav_col2 = grav_col2_0[anomaly_ind];
    grav_row1 = grav_row1_0[anomaly_ind];
    grav_row2 = grav_row2_0[anomaly_ind];
    y_cen = y_cen_0[anomaly_ind]
    x_cen = x_cen_0[anomaly_ind]
    l = l_0[anomaly_ind]
    y = np.array((-l/2,l/2)); 
    
    from_file = {}
    
    if anomaly_ind==0:
        from_file = io.loadmat(nw_grav)
        anomaly_name = 'NW'
    elif anomaly_ind==1:
        from_file = io.loadmat(ne_grav)
        anomaly_name = 'NE'
    elif anomaly_ind==2:
        from_file = io.loadmat(s_grav)
        anomaly_name = 'S'
        
    grav0 = np.squeeze(from_file['grav2'])
    #normalize minimum gravity of region of interest to 0
    grav_mean = np.mean(grav0[:,grav_col1:grav_col2]*1e5,axis=1)
    grav_norm = grav_mean - min(grav_mean[grav_row1:grav_row2])
    
    ################################################################################
    #setting up the coordinate system to calculate gravity from a rectangular prism
    ################################################################################
    
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

    ######################################
    #MCMC iterations
    ######################################

    for run_num in np.arange(1,mcmc_runs+1):
            
        rms_best = 1e10
        
        td = np.random.uniform(20e3,50e3)
        bd = np.random.uniform(td,100e3)
        w = np.random.uniform(50e3,200e3)
       
        td_chain = np.array([]);
        bd_chain = np.array([]);
        w_chain = np.array([]);
        err_chain = np.array([]);
        
        acc = 0
        rej = 0
        
        bd_best = 0
        td_best = 0
        w_best = 0

        for step in np.arange(total_iterations):

            llho, rms1 = likelihood(td, bd, w, grav_norm)
            
            td1 = td+np.random.normal(0,1.5e3)
            bd1 = bd+np.random.normal(0,7e3)
            w1 = w+np.random.normal(0,7e3)
            llhn, rms2 = likelihood(td1, bd1, w1, grav_norm)
            
            judge = np.random.uniform(0, 1)
            
            if w1 <= 0 or td1 <= 0 or td1 >= bd1 or bd1 >= 150e3:
                llhn = 0
                
            test = llhn/llho
            
            if test < judge:
                rej+=1
            else:
                acc+=1
                bd = copy.deepcopy(bd1)
                w = copy.deepcopy(w1)
                td = copy.deepcopy(td1)
                
                bd_chain = np.append(bd_chain,bd)
                w_chain = np.append(w_chain,w)
                td_chain = np.append(td_chain,td)
                err_chain = np.append(err_chain,rms2)
                
                if rms2 < rms_best:
                    rms_best = copy.deepcopy(rms2)
                    w_best = copy.deepcopy(w1)
                    bd_best = copy.deepcopy(bd1)
                    td_best = copy.deepcopy(td1)
                
        
        #####################################################
        #Displaying Results and Saving Results to .mat File
        #####################################################
        
        file_all = 'dike_results_mcmc_' + anomaly_name + '_run' + str(run_num) + '.mat'
        file_bestfit = 'dike_results_mcmc_' + anomaly_name + '_run' + str(run_num) + '_bestfits.mat'
        
        th_best = bd_best-td_best
        th_chain = bd_chain-td_chain
        
        io.savemat(file_all,{'bd_chain':bd_chain,'w_chain':w_chain,'td_chain':td_chain,'acc':acc,'rej':rej,'err_chain':err_chain})            
        io.savemat(file_bestfit,{'td_best':td_best,'w_best':w_best,'bd_best':bd_best,'thh':th_best})
        
        td_avg = np.mean(td_chain)
        bd_avg = np.mean(bd_chain)
        w_avg = np.mean(w_chain)
        th_avg = np.mean(th_chain)

        #The displays show the average +/- the 16% and 84% percentiles

        td_disp = str(r(td_best)/1e3) + ' (' + str(r(td_avg/1e3)) + '+' + str(r(rb(td_chain)-td_avg)/1e3) + ',-' + str(r(td_avg-lb(td_chain))/1e3) + ')'
        bd_disp = str(r(bd_best)/1e3) + ' (' + str(r(bd_avg/1e3)) + '+' + str(r(rb(bd_chain)-bd_avg)/1e3) + ',-' + str(r(bd_avg-lb(bd_chain))/1e3) + ')'
        w_disp = str(r(w_best)/1e3) + ' (' + str(r(w_avg/1e3)) + '+' + str(r(rb(w_chain)-w_avg)/1e3) + ',-' + str(r(w_avg-lb(w_chain))/1e3) + ')'
        th_disp = str(r(th_best)/1e3) + ' (' + str(r(th_avg/1e3)) + '+' + str(r(rb(th_chain)-th_avg)/1e3) + ',-' + str(r(th_avg-lb(th_chain))/1e3) + ')'

        print('Run ' + str(run_num) + ' for Anomaly ' + anomaly_name + ':')
        print('top depth = ' + td_disp)
        print('bottom depth = ' + bd_disp)
        print('width = ' + w_disp)
        print('thickness = ' + th_disp)
        print('best RMS = ' + str(r(rms_best)) + '\n')