# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 14:52:52 2024

@author: thoms
"""

'''
Modifeid BVH traversal to check for occlusion
'''

import numpy as np
from numba import njit

from Hit_functions_fkt import hit_triangle_fkt

@njit
def ray_box_intersection_opt_fkt(ray_origin, divx, divy, divz, box):
    
    if divx >= 0:
        tmin_x = (box[0] - ray_origin[0]) * divx
        tmax_x = (box[1] - ray_origin[0]) * divx
    else:
        tmin_x = (box[1] - ray_origin[0]) * divx
        tmax_x = (box[0] - ray_origin[0]) * divx
    
    if divy >= 0:
        tmin_y = (box[2] - ray_origin[1]) * divy
        tmax_y = (box[3] - ray_origin[1]) * divy
    else:
        tmin_y = (box[3] - ray_origin[1]) * divy
        tmax_y = (box[2] - ray_origin[1]) * divy
    
    if divz >= 0:
        tmin_z = (box[4] - ray_origin[2]) * divz
        tmax_z = (box[5] - ray_origin[2]) * divz
    else:
        tmin_z = (box[5] - ray_origin[2]) * divz
        tmax_z = (box[4] - ray_origin[2]) * divz
    
    t_min = max([tmin_x, tmin_y, tmin_z])
    t_max = min([tmax_x, tmax_y, tmax_z])

    if t_min < t_max:
        return t_min
    else:
        return False

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
def isin_fkt(child_index, visited):
    vis = visited[child_index]
    if vis == child_index:
        return True
    else:
        return False


# return True if line of sight to source
@njit
def occlusion_test_fkt(ray_origin, ray_direction, index_Array, box_node_mat, cen_tri_node_mat, t_max, NTri_in_leafs_vec):
    if ray_direction[0] == 0:
        divx = 1 / (0.0000001)
    else:
        divx = 1 / ray_direction[0]
        
    if ray_direction[1] == 0:
        divy = 1 / (0.0000001)
    else:
        divy = 1 / ray_direction[1]
        
    if ray_direction[2] == 0:
        divz = 1 / (0.0000001)
    else:
        divz = 1 / ray_direction[2]
    
    primary_box = box_node_mat[:,0]
    primary_hit = ray_box_intersection_opt_fkt(ray_origin, divx, divy, divz, primary_box)
    if primary_hit == False:
        return True
    box_node_len = int(len(box_node_mat[0,:]))
    t_vec = np.ones(box_node_len)*(-99)
    stop_number = box_node_len * (-99)
    visited = np.ones(box_node_len)*(-1)
    stack = np.ones(box_node_len)*(-1)
    ones_vec = np.ones(box_node_len)
    stack[0] = 0
    t_vec[0] = primary_hit
    
    #it_counter = 0
    Hit = False
    while Hit == False: 
        #it_counter += 1
        t_min = min_entrie_above_x_fkt(t_vec, -98)
        current_nodes = stack[np.where(t_vec == t_min)[0]]
        current_node = int(max(current_nodes))  
        visited[current_node] = current_node
        stack[current_node] = -1
        t_vec[current_node] = -99
        child1_index = (index_Array[0,current_node])
        child1_index = int(child1_index)
        child2_index = (index_Array[1,current_node])
        child2_index = int(child2_index)
        if child1_index == -9:
            N_tri = int(NTri_in_leafs_vec[child2_index])
            t_tri = np.zeros(N_tri)
            tri_ones_vec = np.ones(N_tri)
            for i in range(N_tri):
                V1 = cen_tri_node_mat[0,:,i,child2_index]
                V2 = cen_tri_node_mat[1,:,i,child2_index]
                V3 = cen_tri_node_mat[2,:,i,child2_index]
                t_tri[i] = hit_triangle_fkt(ray_origin, ray_direction, V1, V2, V3, t_max)
            tri_sum = np.dot(tri_ones_vec, t_tri)
            if tri_sum != 0:
                return False
                    
        else:
            if isin_fkt(child1_index, visited) == False:
                box_1 = box_node_mat[:,child1_index]
                hit_box_1 = ray_box_intersection_opt_fkt(ray_origin, divx, divy, divz, box_1)
                if hit_box_1 != False:
                    stack[child1_index] = child1_index
                    t_vec[child1_index] = hit_box_1
            if isin_fkt(child2_index, visited) == False:
                box_2 = box_node_mat[:,child2_index]
                hit_box_2 = ray_box_intersection_opt_fkt(ray_origin, divx, divy, divz, box_2)
                if hit_box_2 != False:
                    stack[child2_index] = child2_index
                    t_vec[child2_index] = hit_box_2

        if np.dot(t_vec, ones_vec) == stop_number:
            return True