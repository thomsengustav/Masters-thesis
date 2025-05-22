# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:42:45 2025

@author: thoms
"""

'''
graveyard for old BVH functions
'''

import numpy as np
import matplotlib.pyplot as plt
from numba import njit

@njit
def find_second_longest_axis(box):
    xyz_len = np.zeros(3)
    xyz_len[0] = box[1] - box[0]
    xyz_len[1] = box[3] - box[2]
    xyz_len[2] = box[5] - box[4]
    
    max_axis = max(xyz_len)
    index = np.where(xyz_len == max_axis)
    xyz_len[index[0][0]] = 0
    second_max = max(xyz_len)
    if second_max == xyz_len[0]:
        mid_point = box[0] + second_max / 2
        return 0, mid_point
    elif second_max == xyz_len[1]:
        mid_point = box[2] + second_max / 2
        return 1, mid_point
    else:
        mid_point = box[4] + second_max / 2
        return 2, mid_point

@njit
def find_possible_split(centroid_triangle_set):
    cen1 = centroid_triangle_set[3,:,0]
    cen2 = centroid_triangle_set[3,:,1]
    max_diff = 0
    for i in range(3):
        var = abs(cen1[i] - cen2[i]) 
        if var > max_diff:
            max_diff_axis = i
    mid_point = (cen1[max_diff_axis] + cen2[max_diff_axis]) / 2
    high_mat, low_mat, high_counter, low_counter = split_triangles_up_fkt(max_diff_axis, mid_point, centroid_triangle_set)
    
    return high_mat, low_mat, high_counter, low_counter

@njit 
def split_triangles_up_fkt(axis_index, mid_point, centroid_triangle_set):
    N_set = int(len(centroid_triangle_set[0,0,:]))
    
    high_mat = np.zeros((4,3,N_set))
    low_mat = np.zeros((4,3,N_set))
    high_counter = 0
    low_counter = 0
    for i in range(N_set):
        centroid_axis_value = centroid_triangle_set[3,axis_index,i]
        if centroid_axis_value > mid_point:
            high_mat[:,:,high_counter] = centroid_triangle_set[:,:,i]
            high_counter += 1
        else:
            low_mat[:,:,low_counter] = centroid_triangle_set[:,:,i]
            low_counter += 1
    
    high_mat = high_mat[:,:,:high_counter]
    low_mat = low_mat[:,:,:low_counter]
    
    return high_mat, low_mat, high_counter, low_counter

@njit
def ray_box_intersection_fkt(ray_origin, ray_direction, box):
    dir_x = ray_direction[0]
    dir_y = ray_direction[1]
    dir_z = ray_direction[2]
    if dir_x == 0:
        dir_x = 0.0000001
    if dir_y == 0:
        dir_y = 0.0000001
    if dir_z == 0:
        dir_z = 0.0000001
        
    tx0_0 = (box[0] - ray_origin[0]) / dir_x
    tx1_0 = (box[1] - ray_origin[0]) / dir_x
    ty0_0 = (box[2] - ray_origin[1]) / dir_y
    ty1_0 = (box[3] - ray_origin[1]) / dir_y
    tz0_0 = (box[4] - ray_origin[2]) / dir_z
    tz1_0 = (box[5] - ray_origin[2]) / dir_z
    
    tx0 = min([tx0_0, tx1_0])
    tx1 = max([tx0_0, tx1_0])
    ty0 = min([ty0_0, ty1_0])
    ty1 = max([ty0_0, ty1_0])
    tz0 = min([tz0_0, tz1_0])
    tz1 = max([tz0_0, tz1_0])
    
    t_min = max([tx0, ty0, tz0])
    t_max = min([tx1, ty1, tz1])
    #print(t_min)
    #print(t_max)
    if t_min < t_max:
        #print('min')
        return t_min
    else:
        return False

@njit
def find_entries_above_x_fkt(vector, x):
    N = len(vector)
    output = np.ones(N)*(-1)
    counter = 0
    for i in range(N):
        element  = vector[i]
        if element > x:
            output[counter] = element
        counter += 1 
    return output



'''
truly single use test code
'''

def plot_rays_and_fig(ray_o, ray_d, cen_tri_node_mat, xlim, ylim, zlim):
    ax = plt.figure().add_subplot(projection='3d')
    
    # for i in range(6):
    #     V1 = cen_tri_node_mat[0,:,0,i]
    #     V2 = cen_tri_node_mat[1,:,0,i]
    #     V3 = cen_tri_node_mat[2,:,0,i]
    #     ax.plot([V1[0], V2[0], V3[0], V1[0]],[V1[1], V2[1], V3[1], V1[1]],[V1[2], V2[2], V3[2], V1[2]])
    
    ax.plot([ray_o[0], ray_o[0]+ray_d[0]*100],[ray_o[1], ray_o[1]+ray_d[1]*100],[ray_o[2], ray_o[2]+ray_d[2]*100])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    # ax.set_xlim(xlim[0], xlim[1])
    # ax.set_ylim(ylim[0], ylim[1])
    # ax.set_zlim(zlim[0], zlim[1])
    ax.view_init(elev=30, azim=30, roll=0)
    plt.show()

# for i in range(100):
#     ray_origin = Radar_location
#     ray_direction = ray_mat_2[0,:]#np.array([0,-1.,0])
#     plot_rays_and_fig(ray_origin, ray_direction, cen_tri_node_mat, np.array([-3,3]), np.array([-3,3]), np.array([-3,3]))
#     bbb = find_tree_ray_intersection_fkt(ray_origin, ray_direction, index_Array, box_node_mat, cen_tri_node_mat, 200, NTri_in_leafs_vec)
#     print(bbb)

