# -*- coding: utf-8 -*-
"""
Created on Mon Apr 28 13:56:37 2025

@author: thoms
"""

'''
split mesh til Optix
'''

import numpy as np
from better_mesh_fkt import better_mesh_fkt
from load_model_center_rotate import load_big_model

# set max cen-tri distance
max_tri_dist = 0.003

# load model
Triangle_matrix, Triangle_normal_matrix = load_big_model('speed_test_ref_4') 

Triangle_matrix = Triangle_matrix /  0.0254

def rotate_ford(Triangle_matrix):
    N = int(len(Triangle_matrix[0,0,:]))
    new_Triangle_matrix = np.zeros((3,3,N))
    new_Triangle_matrix[:,0,:] = Triangle_matrix[:,0,:]
    new_Triangle_matrix[:,1,:] = Triangle_matrix[:,2,:]
    new_Triangle_matrix[:,2,:] = Triangle_matrix[:,1,:]
    return new_Triangle_matrix

Triangle_matrix = rotate_ford(Triangle_matrix)

def move_geometry_up(triangle_matrix):
    N_triangle = int(len(triangle_matrix[0,0,:]))
    
    # center model xy at 0,0
    x_vec = np.zeros(N_triangle*3)
    y_vec = np.zeros(N_triangle*3)
    z_vec = np.zeros(N_triangle*3)
    for i in range(N_triangle):
        x_point = triangle_matrix[:,0,i]
        x_vec[i:i+3] = x_point
        y_point = triangle_matrix[:,1,i]
        y_vec[i:i+3] = y_point
        z_point = triangle_matrix[:,2,i]
        z_vec[i:i+3] = z_point
    x_avg = np.sum(x_vec)/N_triangle
    y_avg = np.sum(y_vec)/N_triangle
    min_z = min(z_vec)
    
    triangle_matrix[:,0,:] = triangle_matrix[:,0,:] - x_avg
    triangle_matrix[:,1,:] = triangle_matrix[:,1,:] - y_avg
    triangle_matrix[:,2,:] = triangle_matrix[:,2,:] - min_z
    
    return triangle_matrix


Triangle_matrix = better_mesh_fkt(Triangle_matrix, max_tri_dist)

Triangle_matrix = move_geometry_up(Triangle_matrix)


from plot_vector_fkt import plot_mesh_and_normal
# plot_mesh_and_normal(Triangle_matrix[:,:,:50000], Triangle_normal_matrix*0.1, np.array([-5,5]), np.array([-5, 5]), np.array([-0.01, 10]), 30, 30, 0)

def save_optix_trimat(Triangle_matrix):
    N = int(len(Triangle_matrix[0,0,:]))
    N_optix = N*3
    tri_optix = np.zeros((3,N_optix))
    for i in range(N):
        for n in range(3):
            tri_optix[0,i*3+n] = Triangle_matrix[n,0,i]
            tri_optix[1,i*3+n] = Triangle_matrix[n,1,i]
            tri_optix[2,i*3+n] = Triangle_matrix[n,2,i]
    return tri_optix
    
tri_optix = save_optix_trimat(Triangle_matrix)
tri_trans = np.transpose(tri_optix)
np.savetxt('speed_ref_6.txt', tri_trans, delimiter=' ')
