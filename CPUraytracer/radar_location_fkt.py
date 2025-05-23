# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:18:50 2024

@author: thoms
"""

#### find radar location in the scene function

'''
Input: slant_angle (degrees), azimuth_angle (degrees), radar_range (meters)

Output: Radar_location (np.array [x,y,z])

Assumptions: Scene center always located at [0,0,0].
'''

import numpy as np

def radar_location_fkt(slant_angle, azimuth_angle, radar_range):
    
    z_radar_location = np.sin(slant_angle * np.pi / 180) * radar_range
    x_radar_location = np.cos(slant_angle * np.pi / 180) * np.sin(azimuth_angle * np.pi / 180) * radar_range
    y_radar_location = np.cos(slant_angle * np.pi / 180) * np.cos(azimuth_angle * np.pi / 180) * radar_range
    
    Radar_location = np.array([x_radar_location,y_radar_location,z_radar_location])
    return Radar_location

