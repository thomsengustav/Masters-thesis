# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 14:02:03 2025

@author: thoms
"""

'''
BVH functions!
'''

import numpy as np
from numba import njit


@njit
def first_bounding_box_fkt(Triangle_matrix):
    N_triangles = int(len(Triangle_matrix[0,0,:]))
    
    tol = 0.001
    x_vec, y_vec, z_vec = np.zeros(N_triangles*3), np.zeros(N_triangles*3), np.zeros(N_triangles*3)
    for i in range(N_triangles):
        x_point = Triangle_matrix[:,0,i]
        x_vec[i*3:i*3+3] = x_point
        y_point = Triangle_matrix[:,1,i]
        y_vec[i*3:i*3+3] = y_point
        z_point = Triangle_matrix[:,2,i]
        z_vec[i*3:i*3+3] = z_point
    max_x, min_x = max(x_vec), min(x_vec)
    max_y, min_y = max(y_vec), min(y_vec)
    max_z, min_z = max(z_vec), min(z_vec)
    
    box_1 = np.array([min_x - tol, max_x + tol, min_y - tol, max_y + tol, min_z - tol, max_z + tol])
    return box_1


# creates a bounding box for a set of triangles
@njit
def create_bounding_box_fkt(centroid_triangle_set):
    N_set = int(len(centroid_triangle_set[0,0,:]))
    
    tol = 0.001
    x_vec, y_vec, z_vec = np.zeros(N_set*3), np.zeros(N_set*3), np.zeros(N_set*3)
    for i in range(N_set):
        x_point = centroid_triangle_set[0:3,0,i]
        x_vec[i*3:i*3+3] = x_point
        y_point = centroid_triangle_set[0:3,1,i]
        y_vec[i*3:i*3+3] = y_point
        z_point = centroid_triangle_set[0:3,2,i]
        z_vec[i*3:i*3+3] = z_point
    max_x, min_x = max(x_vec), min(x_vec)
    max_y, min_y = max(y_vec), min(y_vec)
    max_z, min_z = max(z_vec), min(z_vec)
    
    box = np.array([min_x - tol, max_x + tol, min_y - tol, max_y + tol, min_z - tol, max_z + tol])
    return box

# finds maximum and minimum number of traignesl pr. leaf
@njit
def tri_in_leaf(tri_node_mat):
    N_leafs = len(tri_node_mat[0,0,0,:])
    number_pr_leaf = np.zeros(N_leafs)
    for i in range(N_leafs):
        number_pr_leaf[i] = len(tri_node_mat[0,0,:,i])
    max_number = max(number_pr_leaf)
    min_number = min(number_pr_leaf)
    
    return min_number, max_number

# checks if the leaf is splittable has more than one triangle.
@njit
def is_splittable(centroid_triangle_mat):
    N_tri = int(len(centroid_triangle_mat[0,0,:]))
    if N_tri == 1:
        return False
    else:
        return True

@njit
def calculate_centroid_fkt(Triangle_matrix):
    N_triangles = int(len(Triangle_matrix[0,0,:]))
    
    centroid_triangle_mat = np.zeros((4,3,N_triangles))
    centroid_triangle_mat[0:3,:,:] = Triangle_matrix
    for i in range(N_triangles):
        V1 = Triangle_matrix[0,:,i]
        V2 = Triangle_matrix[1,:,i]
        V3 = Triangle_matrix[2,:,i]
        Cx = (V1[0] + V2[0] + V3[0]) / 3 
        Cy = (V1[1] + V2[1] + V3[1]) / 3 
        Cz = (V1[2] + V2[2] + V3[2]) / 3 
        centroid_triangle_mat[3,:,i] = np.array([Cx, Cy, Cz])
    
    return centroid_triangle_mat



@njit
def get_subset_fkt(cen_tri_node_mat_i):
    N = len(cen_tri_node_mat_i[0,0,:])
    for i in range(N):
        var = cen_tri_node_mat_i[0,0,i]
        if var == -99:
            return i
    return N


# calculates the longest axis of a bounding box.
# returns axis index and midpoint value of said axis
@njit
def find_longest_axis(box):
    xyz_len = np.zeros(3)
    xyz_len[0] = box[1] - box[0]
    xyz_len[1] = box[3] - box[2]
    xyz_len[2] = box[5] - box[4]
    max_axis = max(xyz_len)
    if max_axis == xyz_len[0]:
        mid_point = box[0] + max_axis / 2
        return 0, mid_point
    elif max_axis == xyz_len[1]:
        mid_point = box[2] + max_axis / 2
        return 1, mid_point
    else:
        mid_point = box[4] + max_axis / 2
        return 2, mid_point

# finds maximum number of triangles in a leaf in cen_tri_node_mat
@njit
def max_tri_in_array(cen_tri_node_mat):
    N_sets = len(cen_tri_node_mat[0,0,0,:])
    max_set_size = 0
    tri_in_set = 0
    for i in range(N_sets):
        tri_in_set = get_subset_fkt(cen_tri_node_mat[:,:,:,i])
        if tri_in_set > max_set_size:
            max_set_size = tri_in_set
    return max_set_size

# takes a subset of triangles and splits them into 2 eaqual size groups aling the axis given by 'input: axis'
@njit
def split_triangles_w_material_fkt(centroid_triangle_set, material_subset, axis):
    N_set = int(len(centroid_triangle_set[0,0,:]))
    var_axis = 9 + axis
    V1_array = centroid_triangle_set[0,:,:]
    V2_array = centroid_triangle_set[1,:,:]
    V3_array = centroid_triangle_set[2,:,:]
    centroid_array = centroid_triangle_set[3,:,:]
    New_array = np.zeros((14,N_set)) # unfolding two matrices (4,3,N_set) and (2,N_set) into (14,N_set)
    New_array[0:3,:] = V1_array 
    New_array[3:6,:] = V2_array 
    New_array[6:9,:] = V3_array 
    New_array[9:12,:] = centroid_array
    New_array[12:14,:] = material_subset
    sorted_cen = New_array[:,New_array[var_axis,:].argsort()] # sort 2d array with respect to centroid coords
    new_centroid_triangle_set = np.zeros((4,3,N_set)) # fold sorted array back to (4,3,N_set)
    new_centroid_triangle_set[0,:,:] = sorted_cen[0:3,:]
    new_centroid_triangle_set[1,:,:] = sorted_cen[3:6,:]
    new_centroid_triangle_set[2,:,:] = sorted_cen[6:9,:]
    new_centroid_triangle_set[3,:,:] = sorted_cen[9:12,:]
    New_material_set = sorted_cen[12:14,:]
    N_half = int(np.floor(N_set/2))
    low_mat = new_centroid_triangle_set[:,:,:N_half]
    low_material = New_material_set[:,:N_half]
    low_counter = N_half
    high_mat = new_centroid_triangle_set[:,:,N_half:]
    high_material = New_material_set[:,N_half:]
    high_counter = int(len(high_mat[0,0,:]))
    return high_mat, low_mat, high_counter, low_counter, high_material, low_material

@njit
def split_triangles_fkt(centroid_triangle_set, axis): # not random direction! longest axis
    N_set = int(len(centroid_triangle_set[0,0,:]))
    var_axis = 9 + axis
    V1_array = centroid_triangle_set[0,:,:]
    V2_array = centroid_triangle_set[1,:,:]
    V3_array = centroid_triangle_set[2,:,:]
    centroid_array = centroid_triangle_set[3,:,:]
    New_array = np.zeros((12,N_set)) # unfolding (4,3,N_set) into (12,N_set)
    New_array[0:3,:] = V1_array 
    New_array[3:6,:] = V2_array 
    New_array[6:9,:] = V3_array 
    New_array[9:12,:] = centroid_array
    sorted_cen = New_array[:,New_array[var_axis,:].argsort()] # sort 2d array with respect to centroid coords
    new_centroid_triangle_set = np.zeros((4,3,N_set)) # fold sorted array back to (4,3,N_set)
    new_centroid_triangle_set[0,:,:] = sorted_cen[0:3,:]
    new_centroid_triangle_set[1,:,:] = sorted_cen[3:6,:]
    new_centroid_triangle_set[2,:,:] = sorted_cen[6:9,:]
    new_centroid_triangle_set[3,:,:] = sorted_cen[9:12,:]
    N_half = int(np.floor(N_set/2))
    low_mat = new_centroid_triangle_set[:,:,:N_half]
    low_counter = N_half
    high_mat = new_centroid_triangle_set[:,:,N_half:]
    high_counter = int(len(high_mat[0,0,:]))
    return high_mat, low_mat, high_counter, low_counter


@njit
def min_entrie_above_x_fkt(vector, x):
    N = len(vector)
    guess = 1000
    for i in range(N):
        element  = vector[i]
        if element > x:
            if guess > element:
                guess = element
    return guess

@njit
def find_entries_below_x_fkt(vector, x):
    N = len(vector)
    output = np.ones(N)*(-99)
    counter = 0
    for i in range(N):
        element  = vector[i]
        if element < x:
            output[counter] = element
        counter += 1 
    return output

@njit
def isin_fkt(child_index, visited):
    vis = visited[child_index]
    if vis == child_index:
        return True
    else:
        return False
    