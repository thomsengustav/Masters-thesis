# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 12:40:06 2025

@author: thoms
"""

'''
New hit_tri_fkt
Möller-trumbore 
'''

from numba import njit
import numpy as np

from fast_vec_operators import cross_fkt, dot_fkt

@njit
def tri_hit_new(ray_dir, ray_org, vertex0, vertex1, vertex2):
    edge1 = vertex1 - vertex0
    edge2 = vertex2 - vertex0 
    dir_edge_cross = cross_fkt(ray_dir, edge2)
    prev_edge_dot = dot_fkt(edge1, dir_edge_cross)
    if prev_edge_dot > -0.00001 and prev_edge_dot < 0.00001:
        return 10000
    f = 1 / prev_edge_dot
    new_vec = ray_org - vertex0
    u = f * dot_fkt(new_vec, dir_edge_cross)
    if u < 0 or u > 1:
        return 10000
    q = cross_fkt(new_vec, edge1)
    v = f * dot_fkt(ray_dir, q)
    if v < 0 or u + v > 1:
        return 10000
    t = f * dot_fkt(edge2, q)
    return t
    

@njit
def hit_triangle_fkt_org(Origin, direction, V1, V2, V3, t_max):
    # calculate matrix elements
    M11, M21, M31 = V1[0] - V2[0], V1[1] - V2[1], V1[2] - V2[2]
    M12, M22, M32 = V1[0] - V3[0], V1[1] - V3[1], V1[2] - V3[2]
    M13, M23, M33 = direction[0], direction[1], direction[2]
    
    E1, E2, E3 = V1[0] - Origin[0],  V1[1] - Origin[1],  V1[2] - Origin[2]
    
    TM1, TM2, TM3 = M22*M33 - M23*M32, M13*M32 - M12*M33, M12*M23 - M22*M13
    Tt1, Tt2, Tt3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    M_denominator = M11*TM1 + M21*TM2 + M31*TM3
    if M_denominator == 0:
        M_denominator = 0.000001
        
    # demon = 1 / M_denominator
    # calculate t
    t_tri = -(M32*Tt1 + M22*Tt2 + M12*Tt3)/M_denominator
    # check if  0 < t < tm
    if t_tri < 0.0000001:
        return 10000
    if t_tri > t_max:
        return 10000
    
    beta = (E1*TM1 + E2*TM2 + E3*TM3)/M_denominator
    
    #check if 0 < beta < 1
    if beta < 0:
        return 10000
    if beta > 1:
        return 10000
    
    TG1, TG2, TG3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    gamma = (M33*TG1 + M23*TG2 + M13*TG3)/M_denominator
    
    # check if gamma > 0 and gamma + beta < 1
    if gamma < 0:
        return 10000
    if gamma > 1 - beta:
        return 10000
    
    return t_tri



@njit
def hit_triangle_fkt(Origin, direction, V1, V2, V3, t_max):
    # calculate matrix elements
    M11, M21, M31 = V1[0] - V2[0], V1[1] - V2[1], V1[2] - V2[2]
    M12, M22, M32 = V1[0] - V3[0], V1[1] - V3[1], V1[2] - V3[2]
    M13, M23, M33 = direction[0], direction[1], direction[2]
    
    E1, E2, E3 = V1[0] - Origin[0],  V1[1] - Origin[1],  V1[2] - Origin[2]
    
    TM1, TM2, TM3 = M22*M33 - M23*M32, M13*M32 - M12*M33, M12*M23 - M22*M13
    Tt1, Tt2, Tt3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    M_denominator = 1 / (M11*TM1 + M21*TM2 + M31*TM3 + 1e-6)
    
    
    # calculate t
    t_tri = -(M32*Tt1 + M22*Tt2 + M12*Tt3) * M_denominator
    # check if  0 < t < tm
    if t_tri < 0.0000001:
        return 10000
    if t_tri > t_max:
        return 10000
    
    beta = (E1*TM1 + E2*TM2 + E3*TM3) * M_denominator
    
    #check if 0 < beta < 1
    if beta < 0:
        return 10000
    if beta > 1:
        return 10000
    
    TG1, TG2, TG3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    gamma = (M33*TG1 + M23*TG2 + M13*TG3) * M_denominator
    
    # check if gamma > 0 and gamma + beta < 1
    if gamma < 0:
        return 10000
    if gamma > 1 - beta:
        return 10000
    
    return t_tri

