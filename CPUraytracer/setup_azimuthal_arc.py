# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:36:25 2024

@author: thoms
"""

#### calculate change in radar location with changes in azimuthal angle

import numpy as np
from numba import njit

# assumes scene center at [0,0,0]

@njit
def azimuth_radar_location_fkt(slant_angle, radar_range,
                               azimuth_steps, azimuth_start, azimuth_end):
    azimuthal_vec =  np.linspace(azimuth_start, azimuth_end, azimuth_steps)
    
    radar_location_mat = np.zeros((azimuth_steps,3))
    
    for i in range(azimuth_steps):
        Azimuthal_angle = azimuthal_vec[i]
        # calculate center of source (target at [0,0,0])
        z_radar = np.sin(slant_angle * np.pi / 180) * radar_range
        x_radar = np.cos(slant_angle * np.pi / 180) * np.sin(Azimuthal_angle * np.pi / 180) * radar_range
        y_radar = np.cos(slant_angle * np.pi / 180) * np.cos(Azimuthal_angle * np.pi / 180) * radar_range
        
        radar_location_mat[i,:] = np.array([x_radar, y_radar, z_radar])
    
    return radar_location_mat

# radar_loc_mat = azimuth_radar_location_fkt(12.8, 100, 3, 0, -2)

import datetime

def save_distance_fkt(distance_vec, RT_name):
    current_time = datetime.datetime.now()
    date = current_time.strftime(" %m%d%Y %H,%M")
    name = RT_name + " " + date + ".txt"
    np.savetxt(name, distance_vec, delimiter=',') 

def save_distance_weight_fkt(distance_vec, weight_vec, RT_name):
    current_time = datetime.datetime.now()
    date = current_time.strftime(" %m%d%Y %H,%M")
    name = RT_name + " " + date + ".txt"
    np.savetxt(name, (distance_vec, weight_vec), delimiter=',') 
    
def calculate_radar_loc(radar_range, slant_angle, Azimuthal_angle):
    z_radar = np.sin(slant_angle * np.pi / 180) * radar_range
    x_radar = np.cos(slant_angle * np.pi / 180) * np.sin(Azimuthal_angle * np.pi / 180) * radar_range
    y_radar = np.cos(slant_angle * np.pi / 180) * np.cos(Azimuthal_angle * np.pi / 180) * radar_range
    
    radar_loc = np.array([x_radar, y_radar, z_radar])
    return radar_loc
