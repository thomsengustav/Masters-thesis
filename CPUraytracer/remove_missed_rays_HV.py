# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 12:26:34 2025

@author: thoms
"""


from numba import njit
import numpy as np
@njit
def remove_missed_rays_HV(distance_vec, weight_HH, weight_HV, weight_VH, weight_VV):
    new_distance_vec = distance_vec[distance_vec != 0]
    N_hits = len(new_distance_vec)
    new_weight_HH = np.zeros((N_hits))
    new_weight_HV = np.zeros((N_hits))
    new_weight_VH = np.zeros((N_hits))
    new_weight_VV = np.zeros((N_hits))
    
    counter = 0
    for i in range(len(distance_vec)):
        if distance_vec[i] != 0:
            new_weight_HH[counter] = weight_HH[i]
            new_weight_HV[counter] = weight_HV[i]
            new_weight_VH[counter] = weight_VH[i]
            new_weight_VV[counter] = weight_VV[i]
            counter += 1
            
    return new_distance_vec, new_weight_HH, new_weight_HV, new_weight_VH, new_weight_VV