# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:58:56 2024

@author: thoms
"""

#### simple ray tracer module


### group intersection code
'''
code for calucating hits with scene objects

input: ray, group

Output: hit_statement (True or False) and t_value (if hit)
'''

import numpy as np
from numba import jit, prange



'''
TODO: bool (hit_sphere, hit_triangle)

Define group, 

'''

# ray vector is given by r = P + t*b, V1, V2, V3 are the vetecies of the triangle
# tm maximum allowed values of t inside the bounding volume.
def hit_triangle_fkt(Origin, direction, V1, V2, V3, t_max):
    # calculate matrix elements
    M11, M21, M31 = V1[0] - V2[0], V1[1] - V2[1], V1[2] - V2[2]
    M12, M22, M32 = V1[0] - V3[0], V1[1] - V3[1], V1[2] - V3[2]
    M13, M23, M33 = direction[0], direction[1], direction[2]
    
    E1, E2, E3 = V1[0] - Origin[0],  V1[1] - Origin[1],  V1[2] - Origin[2]
    
    TM1, TM2, TM3 = M22*M33 - M23*M32, M13*M32 - M12*M33, M12*M23 - M22*M13
    Tt1, Tt2, Tt3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    M_denominator = M11*TM1 + M21*TM2 + M31*TM3
    # calculate t
    t_tri = -(M32*Tt1 + M22*Tt2 + M12*Tt3)/M_denominator
    # check if  0 < t < tm
    if t_tri < 0.00001:
        return False
    if t_tri > t_max:
        return False
    
    beta = (E1*TM1 + E2*TM2 + E3*TM3)/M_denominator
    
    #check if 0 < beta < 1
    if beta < 0:
        return False
    if beta > 1:
        return False
    
    TG1, TG2, TG3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    gamma = (M33*TG1 + M23*TG2 + M13*TG3)/M_denominator
    
    # check if gamma > 0 and gamma + beta < 1
    if gamma < 0:
        return False
    if gamma > 1 - beta:
        return False
    
    return t_tri


def hit_sphere_fkt(Origin, direction, centrum, Radius):
    new_vec = Origin - centrum
    k1, k2, k3 = np.dot(direction,direction), np.dot(direction, new_vec), np.dot(new_vec, new_vec) - Radius**2
    disc = k2**2 - k1 * k3
    if disc < 0:
        return  False
    else:
        new_const = np.sqrt(disc)
        t_min, t_pos = (-k2 - new_const) / k1, (-k2 + new_const) / k1
        return min(t_min, t_pos)

def hit_detector_fkt(Origin, direction, centrum, Radius, last_hit):
    new_vec = Origin - centrum
    k1, k2, k3 = np.dot(direction, direction), np.dot(direction, new_vec), np.dot(new_vec, new_vec) - Radius**2
    disc = k2**2 - k1 * k3
    if disc < 0:
        return  False
    else:
        new_vec = last_hit - centrum
        distance = np.sqrt(new_vec[0]**2 + new_vec[1]**2 + new_vec[2]**2)
        return distance



