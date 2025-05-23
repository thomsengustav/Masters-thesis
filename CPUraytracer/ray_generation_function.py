# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 19:09:46 2024

@author: thoms
"""

#### Ray generation function!
"""
Inputs: number_of_rays (integer), rotation_matrix (3x3 np.array), beam_spread (integer)

Output: Vector components of the launced ray directions , x, y, z as np.arrays

Assumptions: Scene center always located at [0,0,0]. Standard offset from source at 1 meter.
"""

from rotation_matrix_fkt import beam_direction_fkt

import numpy as np
from numba import njit

@njit
def launch_ray_components_fkt(number_of_rays, beam_spread, rotation_matrix, Radar_location):
    x, y = np.random.normal(0, beam_spread, number_of_rays), np.random.normal(0, beam_spread, number_of_rays)
    z = np.zeros(number_of_rays)
    
    # preform rotation
    x_rot = x * rotation_matrix[0,0] + y * rotation_matrix[0,1] + z * rotation_matrix[0,2]
    y_rot = x * rotation_matrix[1,0] + y * rotation_matrix[1,1] + z * rotation_matrix[1,2]
    z_rot = x * rotation_matrix[2,0] + y * rotation_matrix[2,1] + z * rotation_matrix[2,2]
    
    Displacement_vector = Radar_location + 10*beam_direction_fkt(Radar_location)
    x = x_rot + Displacement_vector[0]
    y = y_rot + Displacement_vector[1]
    z = z_rot + Displacement_vector[2]
    
    # calculate normalized vector components
    x_ray_component = x - Radar_location[0]
    y_ray_component = y - Radar_location[1] 
    z_ray_component = z - Radar_location[2] 
    
    vector_length_inverse = 1 / np.sqrt(x_ray_component**2 + y_ray_component**2 + z_ray_component**2 )
    
    x_ray_component_norm = np.multiply(x_ray_component, vector_length_inverse)  
    y_ray_component_norm = np.multiply(y_ray_component, vector_length_inverse)
    z_ray_component_norm = np.multiply(z_ray_component, vector_length_inverse)
    
    return x_ray_component_norm, y_ray_component_norm, z_ray_component_norm


@njit
def launch_ray_components_uniform_fkt(number_of_rays, beam_size, rotation_matrix, Radar_location):
    x, y = (np.random.rand(number_of_rays) - 0.5)*beam_size,  (np.random.rand(number_of_rays) - 0.5)*beam_size
    z = np.zeros(number_of_rays)
    
    # preform rotation
    x_rot = x * rotation_matrix[0,0] + y * rotation_matrix[0,1] + z * rotation_matrix[0,2]
    y_rot = x * rotation_matrix[1,0] + y * rotation_matrix[1,1] + z * rotation_matrix[1,2]
    z_rot = x * rotation_matrix[2,0] + y * rotation_matrix[2,1] + z * rotation_matrix[2,2]
    
    Displacement_vector = Radar_location + 35*beam_direction_fkt(Radar_location)
    x = x_rot + Displacement_vector[0]
    y = y_rot + Displacement_vector[1]
    z = z_rot + Displacement_vector[2]
    
    # calculate normalized vector components
    x_ray_component = x - Radar_location[0]
    y_ray_component = y - Radar_location[1] 
    z_ray_component = z - Radar_location[2] 
    
    vector_length_inverse = 1 / np.sqrt(x_ray_component**2 + y_ray_component**2 + z_ray_component**2 )
    
    x_ray_component_norm = np.multiply(x_ray_component, vector_length_inverse)  
    y_ray_component_norm = np.multiply(y_ray_component, vector_length_inverse)
    z_ray_component_norm = np.multiply(z_ray_component, vector_length_inverse)
    
    return x_ray_component_norm, y_ray_component_norm, z_ray_component_norm


