# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 11:46:50 2025

@author: thoms
"""

'''
add roughness to mesh based on roughness_par
roughness_par controls the spread of vertex displacements
'''

import numpy as np
from numba import njit

from better_mesh_fkt import calculate_normals
from fast_vec_operators import len_fkt

# Compute the average length of triangle
@njit    
def get_avg_side_len(vertices):
    # get xyz of each vertex
    V1 = vertices[0,:]
    V2 = vertices[1,:]
    V3 = vertices[2,:]
    # compute avg. length
    len1 = len_fkt(V1-V2)
    len2 = len_fkt(V1-V3)
    len3 = len_fkt(V3-V2)
    avg_len = (len1+len2+len3)/3
    return avg_len


# compute displacements vectors by scaling the normal vector components
@njit
def get_displacement_xyz(normal, displacement_vec):
    dis_xyz = np.zeros((3,3))
    for i in range(3):
        dis_xyz[i,0] = normal[0] * displacement_vec[i]
        dis_xyz[i,1] = normal[1] * displacement_vec[i]
        dis_xyz[i,2] = normal[2] * displacement_vec[i]
    return dis_xyz


def add_plane_roughness(Triangle_matrix, roughness_par):
    # add random displacement to vertices perpendicular to triangle plane
    # roughness_par controls spread of random gausian for displacement
    N = int(len(Triangle_matrix[0,0,:]))
    new_tri_mat = np.zeros((3,3,N))
    normals = calculate_normals(Triangle_matrix)
    for i in range(N):
        avg_len = get_avg_side_len(Triangle_matrix[:,:,i])
        displacement_vec = np.random.normal(0, roughness_par, 3)*avg_len
        dis_xyz = get_displacement_xyz(normals[i,:], displacement_vec)
        # add displacement to original triangle matrix
        for n in range(3):
            for m in range(3):
                new_tri_mat[n,m,i] = dis_xyz[n,m] + Triangle_matrix[n,m,i]
    return new_tri_mat



