# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:45:29 2025

@author: thoms
"""

'''
remove missed rays
'''

from numba import njit
import numpy as np

# assumes detector location = radar location
@njit
def remove_missed_rays(distance_vec, weight_vec):
    new_distance_vec = distance_vec[distance_vec != 0]
    N_hits = len(new_distance_vec)
    new_weight_vec = np.zeros((N_hits))
    
    counter = 0
    for i in range(len(distance_vec)):
        if distance_vec[i] != 0:
            new_weight_vec[counter] = weight_vec[i]
            counter += 1
            
    return new_distance_vec, new_weight_vec

