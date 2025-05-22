# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:58:56 2024

@author: thoms
"""

#### Hit objects functions

'''
Functions for calculating hits of scene objects
'''

import numpy as np
from numba import njit, float64


# ray vector is given by r = P + t*b, V1, V2, V3 are the vetecies of the triangle
# tm maximum allowed values of t inside the bounding volume.
@njit
def hit_triangle_fkt(Origin, direction, V1, V2, V3, t_max):
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
    # calculate t
    t_tri = -(M32*Tt1 + M22*Tt2 + M12*Tt3)/M_denominator
    # check if  0 < t < tm
    if t_tri < 0.0000001:
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

@njit
def hit_triangle_par_fkt(Origin, direction, V1_mat, V2_mat, V3_mat, t_max):
    V1x, V1y, V1z = V1_mat[0,:], V1_mat[1,:], V1_mat[2,:]
    V2x, V2y, V2z = V2_mat[0,:], V2_mat[1,:], V2_mat[2,:]
    V3x, V3y, V3z = V3_mat[0,:], V3_mat[1,:], V3_mat[2,:]

    M13, M23, M33 = direction[0], direction[1], direction[2]
    E1, E2, E3 = V1x - Origin[0],  V1y - Origin[1],  V1z - Origin[2]
    M11, M21, M31 = V1x - V2x, V1y - V2y, V1z - V2z
    M12, M22, M32 = V1x - V3x, V1y - V3y, V1z - V3z

    TM1, TM2, TM3 = M22*M33 - M23*M32, M13*M32 - M12*M33, M12*M23 - M22*M13
    Tt1, Tt2, Tt3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    test = (M11*TM1 + M21*TM2 + M31*TM3)
    inv_denominator = 1 /  (M11*TM1 + M21*TM2 + M31*TM3)

    t_tri = -(M32*Tt1 + M22*Tt2 + M12*Tt3) * inv_denominator
    
    t_tri[t_tri < 0] = 0
    t_tri[t_tri > t_max] = 0

    beta = (E1*TM1 + E2*TM2 + E3*TM3) * inv_denominator

    #check if 0 < beta < 1
    beta_index1 = np.where(beta < 0)
    t_tri[beta_index1[0]] = 0
    beta_index2 = np.where(beta > 1)
    t_tri[beta_index2[0]] = 0

    TG1, TG2, TG3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    gamma = (M33*TG1 + M23*TG2 + M13*TG3) * inv_denominator

    gamma_index1 = np.where(gamma < 0)
    t_tri[gamma_index1[0]] = 0
    gamma_index2 = np.where(gamma > 1 - beta)
    t_tri[gamma_index2[0]] = 0

    tri_hits = t_tri[t_tri != 0]
    if int(len(tri_hits)) == 0:
        return False, False
    else:
        min_t = np.min(tri_hits)
        tri_index = np.where(t_tri == min_t)
        tri_index = tri_index[0][0]
        return min_t, tri_index
    
    
@njit
def divide_mesh_fkt(Triangle_matrix, M):
    N_tot = int(len(Triangle_matrix[0,0,:]))
    m_it = int(np.ceil(N_tot / M))
    rest_len = N_tot - (m_it-1)*M
    V1_super_mat = np.zeros((3,M, m_it))
    V2_super_mat = np.zeros((3,M, m_it))
    V3_super_mat = np.zeros((3,M, m_it))
    for i in range(m_it-1):
        V1_super_mat[:,:,i] = Triangle_matrix[0,:,M*i:(i+1)*M]
        V2_super_mat[:,:,i] = Triangle_matrix[1,:,M*i:(i+1)*M]
        V3_super_mat[:,:,i] = Triangle_matrix[2,:,M*i:(i+1)*M]
    V1_super_mat[:,:rest_len,m_it-1] = Triangle_matrix[0,:,(m_it-1)*M:]
    V2_super_mat[:,:rest_len,m_it-1] = Triangle_matrix[1,:,(m_it-1)*M:]
    V3_super_mat[:,:rest_len,m_it-1] = Triangle_matrix[2,:,(m_it-1)*M:]
    return V1_super_mat, V2_super_mat, V3_super_mat, m_it, rest_len

@njit
def hit_triangle_par_div_fkt(Origin, direction, V1_super_mat, V2_super_mat, V3_super_mat, m_it, M, rest_len, t_max):

    prev_min_t, tri_index = hit_triangle_par_fkt(Origin, direction, V1_super_mat[:,:rest_len,-1], V2_super_mat[:,:rest_len,-1], V3_super_mat[:,:rest_len,-1], t_max)
    final_tri_index = (m_it-1) * M + tri_index
    
    for i in range(m_it-1):
        min_t, tri_index = hit_triangle_par_fkt(Origin, direction, V1_super_mat[:,:,i], V2_super_mat[:,:,i], V3_super_mat[:,:,i], t_max)
        
        if min_t < prev_min_t:
            prev_min_t = min_t
            final_tri_index = i*M + tri_index
    return prev_min_t, final_tri_index



@njit
def hit_sphere_fkt(Origin, direction, centrum, Radius):
    new_vec = Origin - centrum
    k1, k2, k3 = np.dot(direction,direction), np.dot(direction, new_vec), np.dot(new_vec, new_vec) - Radius**2
    disc = k2**2 - k1 * k3
    if disc < 0:
        return  False
    else:
        new_const = np.sqrt(disc)
        t_min, t_pos = (-k2 - new_const) / k1, (-k2 + new_const) / k1
        t = min(t_min, t_pos)
        if t > 0:
            return t
        else:
            return False

@njit
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

@njit
def hit_plane_fkt(Origin, direction, point, normal, t_max):
    t = np.dot(normal, point - Origin) / np.dot(normal, direction)
    if t > t_max or t < 0:
        return False
    else:
        return t


