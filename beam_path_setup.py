# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 09:25:43 2025

@author: thoms
"""

'''
script for calculating and plotting the beam path of the THz setup 
as a function of slant angle.
'''

import numpy as np
import matplotlib.pyplot as plt

# start and end angle of the setup in degrees
theta_0 = 0
theta_1 = 90
N_theta = 20

# stage position and length
stage_x = 20
stage_y = 34
stage_l = 5

# offset of emitter from rail in cm. Assume parallel beam with the rail
offset = 5
# rail length
rail_L = 112

theta = np.linspace(theta_0,theta_1,N_theta)


beam_mat = np.zeros((6,N_theta))
for i in range(N_theta):
    theta_pi = theta[i] * np.pi / 180
    rail_y = rail_L * np.sin(theta_pi)
    rail_x = rail_L * np.cos(theta_pi)
    
    theta_2pi = theta_pi * 2
    
    offset_x = -offset * np.sin(theta_pi)
    offset_y = offset * np.cos(theta_pi)
    
    beam_x = offset_x + rail_x
    beam_y = offset_y + rail_y
    
    start_x = offset_x
    start_y = offset_y
    beam_mat[0,i] = start_x
    beam_mat[1,i] = start_y
    beam_mat[2,i] = beam_x
    beam_mat[3,i] = beam_y
    beam_mat[4,i] = rail_x
    beam_mat[5,i] = rail_y


sx = np.array([stage_x + stage_l,stage_x - stage_l])
sy = np.array([stage_y,stage_y])
for i in range(N_theta):
    rx = np.array([0, beam_mat[4,i]])
    ry = np.array([0, beam_mat[5,i]])
    plt.plot(rx,ry,'k')
    
for i in range(N_theta):
    x = np.array([beam_mat[0,i], beam_mat[2,i]])
    y = np.array([beam_mat[1,i], beam_mat[3,i]])
    
    plt.plot(x,y,'b')
    
plt.plot(sx,sy,'r')  
plt.xlim([0,rail_L])
plt.ylim([0,rail_L])
plt.axis('equal')
plt.show()


