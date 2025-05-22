# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:11:03 2024

@author: thoms
"""


# faster crossproduct and vector norm for 3D vectors

import numpy as np
from numba import njit

'''
fast vector operation functions on 3x1 vectors!
speed inrease is measured as Speed_np / Speed_new, where Speed_np is 
the time for numpy function and Speed_new is the time for the new function
'''

# 1.33 times faster
@njit
def cross_fkt(v0,v1): # was tested to be faster than np.cross for this application
    x = v0[1]*v1[2]-v0[2]*v1[1]
    y = -v0[0]*v1[2]+v0[2]*v1[0]
    z = v0[0]*v1[1]-v0[1]*v1[0]
    return np.array([x,y,z])

# 3.73 times faster
@njit
def len_fkt(vec): # was tested to be faster than np.linalg.norm for this application
    # calculates lenght of 3d vector   
    inside = vec[0]**2 + vec[1]**2 + vec[2]**2
    return np.sqrt(inside)

@njit
def norm_vec_fkt(vec): # input vec - output normalized vec
    size_vec = len_fkt(vec)
    return vec / size_vec

# 1.18 times faster
@njit
def dot_fkt(v0,v1): 
    return v0[0]*v1[0] + v0[1]*v1[1] + v0[2]*v1[2]

# 2.00 times faster
@njit
def mat_vec_prod(mat, vec):
    x = mat[0,0]*vec[0] + mat[0,1]*vec[1] + mat[0,2]*vec[2]
    y = mat[1,0]*vec[0] + mat[1,1]*vec[1] + mat[1,2]*vec[2]
    z = mat[2,0]*vec[0] + mat[2,1]*vec[1] + mat[2,2]*vec[2]
    return np.array([x,y,z])

