# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 09:05:54 2024

@author: thoms
"""

#### loadmodel and calclate normals!
import numpy as np

def distance_xy(mat):
    N = int(len(mat[:,0,0]))
    x_vec = np.zeros(N*3)
    y_vec = np.zeros(N*3)
    for i in range(N):
        x_point = mat[i,:,0]
        x_vec[i:i+3] = x_point
        y_point = mat[i,:,1]
        y_vec[i:i+3] = y_point
    
    min_x = min(x_vec)
    max_x = max(x_vec)
    distance_x = max_x - min_x
    min_y = min(y_vec)
    max_y = max(y_vec)
    distance_y = max_y - min_y
    return distance_x, distance_y


def load_model_kampvogn_vec_scene_centrum(name):
    model_mat = np.load(name + '.npy')
    # rescale to meter
    triangle_matrix = model_mat * 0.0254
    
    N_triangle = int(len(triangle_matrix[:,0,0]))
    
    # center model xy at 0,0
    x_vec = np.zeros(N_triangle*3)
    y_vec = np.zeros(N_triangle*3)
    z_vec = np.zeros(N_triangle*3)
    for i in range(N_triangle):
        x_point = triangle_matrix[i,:,0]
        x_vec[i:i+3] = x_point
        y_point = triangle_matrix[i,:,1]
        y_vec[i:i+3] = y_point
        z_point = triangle_matrix[i,:,2]
        z_vec[i:i+3] = z_point
    x_avg = np.sum(x_vec)/N_triangle
    y_avg = np.sum(y_vec)/N_triangle
    min_z = min(z_vec)
    
    triangle_matrix[:,:,0] = triangle_matrix[:,:,0] - x_avg
    triangle_matrix[:,:,1] = triangle_matrix[:,:,1] - y_avg
    triangle_matrix[:,:,2] = triangle_matrix[:,:,2] - min_z
    
    return triangle_matrix


def calculate_normals(tri_mat):
    N_triangle = int(len(tri_mat[:,0,0]))
    triangle_normal_matrix = np.zeros((N_triangle,3))
    for i in range(N_triangle):
        V1 = tri_mat[i,0,:]
        V2 = tri_mat[i,1,:]
        V3 = tri_mat[i,2,:]

        norm = np.cross(V1-V3,V2-V3)
        norm = norm / np.linalg.norm(norm)
        triangle_normal_matrix[i,:] = norm
    return triangle_normal_matrix



def reshape_array(tri_mat):
    N_triangle = int(len(tri_mat[:,0,0]))
    new_tri_mat = np.zeros((3,3,N_triangle))
    
    for i in range(N_triangle):
        V1 = tri_mat[i,0,:]
        V2 = tri_mat[i,1,:]
        V3 = tri_mat[i,2,:]
        new_tri_mat[:,:,i] = np.array([V1,V2,V3])
    return new_tri_mat



def load_big_model(name):
    triangle_matrix = load_model_kampvogn_vec_scene_centrum(name)
    triangle_normal_matrix = calculate_normals(triangle_matrix)
    new_triangle_matrix = reshape_array(triangle_matrix)
    
    return new_triangle_matrix, triangle_normal_matrix
