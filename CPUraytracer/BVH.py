# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 13:38:46 2024

@author: thoms
"""

#### Bounding volumes!

import numpy as np
from point_scater_triangle import Point_scatter_fkt
from numba import njit
from load_gmesh_square import load_gmesh_triangle_model
import matplotlib.pyplot as plt

from fast_vec_operators import cross_fkt, len_fkt, norm_vec_fkt

from BVH_library_funcs import first_bounding_box_fkt, create_bounding_box_fkt, calculate_centroid_fkt, find_longest_axis
from BVH_library_funcs import split_triangles_fkt, tri_in_leaf, is_splittable, get_subset_fkt
from BVH_library_funcs import max_tri_in_array

'''
old functions that can be found in BVH_old_fkts: find_second_longest_axis, find_possible_split, 
split_triangles_up_fkt, ray_box_intersection_fkt, 
'''


# BVH builder. Takes mesh as Triangle_matrix and computes the BVH structure of AABBs (box)
# with max_in_leaf controlling the number of maximum triangles allowed in the final BVH structures leaves.
# Output: box_node_mat - ordered array of all AABB in the BVH. From left to right the boxes get deeper into the BVH tree.
# cen_tri_node_mat - new ordering of triangles grouped in the leaves of the BVH
# parent_vec - notes possible no_splits i.e. if a leaf cannot be split as it only has one triangle
# (not really a problem as the current split method splits evenly and we ussually have a couple hundred triangles pr. leaf) 
# N_box_vec - notes the lenght of box_node_mat efter each iteration 
def BVH_builder(Triangle_matrix, max_in_leaf):
    primary_box = first_bounding_box_fkt(Triangle_matrix)
    box_node_mat = np.zeros((6,1))
    box_node_mat[:,0] = primary_box
    
    centroid_triangle_mat = calculate_centroid_fkt(Triangle_matrix)
    N_tri = len(centroid_triangle_mat[0,0,:])
    cen_tri_node_mat = np.zeros((4,3,N_tri,1))
    cen_tri_node_mat[:,:,:,0] =  centroid_triangle_mat
    
    min_tri_in_leaf, max_tri_in_leaf = tri_in_leaf(cen_tri_node_mat)
    N_box_prev = 0
    global_no_split_vec = np.zeros(30)
    parent_vec = np.ones((30,len(cen_tri_node_mat[0,0,:,0])))*(-1)
    N_box_vec = np.zeros(30)
    iteration_counter = 0
    
    while min_tri_in_leaf > max_in_leaf:
        N_parents = len(cen_tri_node_mat[0,0,0,:])
        N_box = len(box_node_mat[0,:])
        max_N_tri = len(cen_tri_node_mat[0,0,:,0])
        cen_tri_node_mat_new = np.ones((4,3,max_N_tri,N_parents*2))*(-99)
        box_node_mat_new = np.zeros((6,N_parents*2 + N_box))
        box_node_mat_new[:,:(N_box)] = box_node_mat
        no_split_counter = 0
        N_box_vec[iteration_counter] = N_box
        for i in range(N_parents):
            subset_N = get_subset_fkt(cen_tri_node_mat[:,:,:,i])
            cen_tri_subset = cen_tri_node_mat[:,:,:subset_N,i]
            
            box_subset = box_node_mat[:,N_box - N_box_prev + i - 1]
            split_var = is_splittable(cen_tri_subset)
            if split_var == False:
                cen_tri_node_mat_new[:,:,0,i*2 - no_split_counter] = cen_tri_subset[:,:,0]
                Box = create_bounding_box_fkt(cen_tri_subset)
                box_node_mat_new[:,N_box + i*2 - no_split_counter] = Box
                no_split_counter += 1
                global_no_split_vec[iteration_counter] += 1
                parent_vec[iteration_counter, i] = i
                continue
            
            axis_index, midpoint = find_longest_axis(box_subset)
            
            M1, M2, C1, C2 = split_triangles_fkt(cen_tri_subset, axis_index)
        
            cen_tri_node_mat_new[:,:,:C1,i*2 - no_split_counter] = M1
            cen_tri_node_mat_new[:,:,:C2,(i+1)*2-1 - no_split_counter] = M2
            
            Box1 = create_bounding_box_fkt(M1)
            Box2 = create_bounding_box_fkt(M2)
            
            box_node_mat_new[:,N_box + i*2 - no_split_counter] = Box1
            box_node_mat_new[:,N_box + (i+1)*2-1 - no_split_counter] = Box2
        iteration_counter += 1
        N_box_prev = N_box
        
        box_node_mat = box_node_mat_new[:,:(len(box_node_mat_new[0,:])-no_split_counter)]
        
        cen_tri_node_mat = cen_tri_node_mat_new[:,:,:,:(N_parents*2 - no_split_counter)]
        max_tri_in_set = max_tri_in_array(cen_tri_node_mat)
        cen_tri_node_mat = cen_tri_node_mat[:,:,:max_tri_in_set,:]
        min_tri_in_leaf, max_tri_in_leaf = tri_in_leaf(cen_tri_node_mat)
        print(min_tri_in_leaf) 
        
    return box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec

# create index array of the BVH
def tree_node_fkt(box_node_mat, cen_tri_node_mat, N_box_vec, parent_vec):
    N_box_last = len(box_node_mat[0,:])
    index_last_number = np.where(N_box_vec == 0)
    N_box_vec_tot = N_box_vec[:1+index_last_number[0][0]]
    N_box_vec_tot[-1] = N_box_last
    
    iteration = len(N_box_vec_tot)
    global_index = 1
    counter = 0
    index_array = np.zeros((2,N_box_last))
    N_diff_vec = np.zeros(iteration)
    N_diff_vec[0] = N_box_vec_tot[0]
    for i in range(iteration-1):
        N_diff_vec[i+1] = N_box_vec_tot[i+1] - N_box_vec_tot[i]
    print(N_diff_vec)
    for i in range(iteration-1):
        for n in range(int(N_diff_vec[i])):
            
            no_split = parent_vec[i,n]
            if no_split != -1:
                child2_index = global_index
                global_index += 1
                child1_index = -1
            else:
                child1_index = global_index
                global_index += 1
                child2_index = global_index
                global_index += 1
            index_array[0,counter] = child1_index
            index_array[1,counter] = child2_index
            counter += 1
    
    # assign leafs to their triangle!
    for i in range(len(cen_tri_node_mat[0,0,0,:])):
        index_array[0,counter] = -9
        index_array[1,counter] = i
        counter += 1
    return index_array, N_diff_vec

def follow_to_leaf(index_array, branch_index):
    leaf_check = -1
    index = branch_index
    while leaf_check != -9:
        new_index = int(index_array[1,index])
        leaf_check = index_array[0,index]
        index = new_index
    tri_index = new_index

    return tri_index

# optimize BVH index array
def optimize_index_array(index_Array, box_node_mat, cen_tri_node_mat, N_diff_vec):
    New_array = index_Array
    N_tri_moved = 0
    visited = np.ones(len(box_node_mat[0,:]))*(-1)
    stack = np.ones(len(box_node_mat[0,:]))*(-1)
    stack[0] = 0

    while N_tri_moved < N_diff_vec[-2]:
        current_node = int(min(n for n in stack  if n>-0.1))
        visited[current_node] = current_node
        stack[current_node] = -1
        
        child1_index = int(index_Array[0,current_node])
        child2_index = int(index_Array[1,current_node])
        if child1_index == -1:
            tri_index = follow_to_leaf(index_Array, child2_index)
            New_array[1,current_node] = tri_index
            New_array[0,current_node] = -9
            N_tri_moved += 1
        elif child1_index == -9:
            N_tri_moved += 1
        else:
            if np.isin(child1_index, visited) == False:
                stack[child1_index] = child1_index
            if np.isin(child2_index, visited) == False:
                stack[child2_index] = child2_index
    return New_array



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

from Hit_functions_fkt import hit_triangle_fkt
from BVH_library_funcs import min_entrie_above_x_fkt, find_entries_below_x_fkt, isin_fkt

# get number of triangles in leaf
@njit
def find_tri_in_leafs(cen_tri_node_mat):
    N_tri = int(len(cen_tri_node_mat[0,0,:,0]))
    N_leafs = int(len(cen_tri_node_mat[0,0,0,:]))
    NTri_in_leafs_vec = np.zeros(N_leafs)
    for i in range(N_leafs):
        counter = 0
        for n in range(N_tri):
            if cen_tri_node_mat[0,0,n,i] != -99:
                counter += 1
        NTri_in_leafs_vec[i] = counter
    return NTri_in_leafs_vec

# BVH traversal function
@njit
def find_tree_ray_intersection_fkt(ray_origin, ray_direction, index_Array, box_node_mat, cen_tri_node_mat, t_max, NTri_in_leafs_vec):
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
        return False, False, False
    box_node_len = int(len(box_node_mat[0,:]))
    t_vec = np.ones(box_node_len)*(-99)
    stop_number = box_node_len * (-99)
    visited = np.ones(box_node_len)*(-1)
    stack = np.ones(box_node_len)*(-1)
    ones_vec = np.ones(box_node_len)
    stack[0] = 0
    t_vec[0] = primary_hit
    
    best_t_tri = t_max
    best_child2_index = -1
    best_tri_index = -1
    
    Hit = False
    while Hit == False: 
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
                tri_min = min_entrie_above_x_fkt(t_tri, 0)
                if best_t_tri > tri_min:
                    best_t_tri = tri_min
                    best_child2_index = child2_index
                    tri_index = np.where(t_tri == best_t_tri)
                    best_tri_index = int(tri_index[0][0])
                    t_vec = find_entries_below_x_fkt(t_vec, best_t_tri)
                    
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
            #print(it_counter)
            if best_t_tri == t_max:
                return False, False, False
            else:
                return best_t_tri, best_child2_index, best_tri_index


def tri_normal_calculator_BVH_fkt(cen_tri_node_mat):
    N_leaf = int(len(cen_tri_node_mat[0,0,0,:]))
    Max_tri = int(len(cen_tri_node_mat[0,0,:,0]))
    BVH_tri_normal_mat = np.zeros((3, Max_tri, N_leaf))
    for i in range(N_leaf):
        
        for n in range(Max_tri):
            V1 = cen_tri_node_mat[0,:,n,i]
            V2 = cen_tri_node_mat[1,:,n,i]
            V3 = cen_tri_node_mat[2,:,n,i]
            if V1[0] == -99:
                continue
            else:
                norm1 = cross_fkt(V1-V3,V2-V3)
                BVH_tri_normal_mat[:,n,i] = norm_vec_fkt(norm1)
    return BVH_tri_normal_mat

#### functions for testing and plotting BVH 
def plot_bounding_volume(index_array, box_node_mat, cen_tri_node_mat, N_diff_vec, xlim_v, ylim_v, zlim_v, elevation_angle, azimuth_angle, roll_angle):
    len_vec = len(N_diff_vec)
    N_accum=0
    for i in range(len_vec):
        ax = plt.figure().add_subplot(projection='3d')

        for n in range(int(N_diff_vec[i])):
            x_min = box_node_mat[0,n+N_accum]
            x_max = box_node_mat[1,n+N_accum]
            y_min = box_node_mat[2,n+N_accum]
            y_max = box_node_mat[3,n+N_accum]
            z_min = box_node_mat[4,n+N_accum]
            z_max = box_node_mat[5,n+N_accum]
            
            x_vec = [x_min, x_max, x_max, x_min, x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_max, x_max, x_min, x_min]
            y_vec = [y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_max, y_max]
            z_vec = [z_min, z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max, z_max, z_max, z_min, z_min, z_max, z_max, z_min]
            ax.plot(x_vec ,y_vec, z_vec)
            
        ax.set_xlim([xlim_v[0], xlim_v[1]])
        ax.set_ylim([ylim_v[0], ylim_v[1]])
        ax.set_zlim([zlim_v[0], zlim_v[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=elevation_angle, azim=azimuth_angle, roll=roll_angle)
        plt.show()
        N_accum += int(N_diff_vec[i])

# plot_bounding_volume(new_index_array, box_node_mat, cen_tri_node_mat, N_diff_vec, np.array([-2, 2]), np.array([-2, 2]), np.array([0, 4]), 30, 60, 0)



@njit
def find_tri_in_box(box_index, new_index_array, NTri_in_leafs_vec, cen_tri_node_mat):
    array_len = int(len(new_index_array[0,:]))
    stack = np.ones(array_len+1)*(999999)
    visited = np.ones(array_len)*(-1)
    stack[0] = new_index_array[0,box_index]
    stack[1] = new_index_array[1,box_index]
    tri_in_box = find_N_tri_in_box(box_index, new_index_array, NTri_in_leafs_vec)
    new_cen_tri_node_mat = np.zeros((4,3,int(tri_in_box)))
    current_node = 0
    tri_counter = 0
    while current_node != 999999:
        current_node = int(np.min(stack))
        if current_node == 999999:
            return new_cen_tri_node_mat
        stack_index = np.where(stack == current_node)[0][0]
        visited[current_node] = current_node
        stack[stack_index] = 999999
        child1_index = (new_index_array[0,current_node])
        child1_index = int(child1_index)
        child2_index = (new_index_array[1,current_node])
        child2_index = int(child2_index)
        if child1_index == -9:
            N_tri = int(NTri_in_leafs_vec[child2_index])
            t_tri = np.zeros(N_tri)
            for i in range(N_tri):
                new_cen_tri_node_mat[0,:,tri_counter] = cen_tri_node_mat[0,:,i,child2_index]
                new_cen_tri_node_mat[1,:,tri_counter] = cen_tri_node_mat[1,:,i,child2_index]
                new_cen_tri_node_mat[2,:,tri_counter] = cen_tri_node_mat[2,:,i,child2_index]
                new_cen_tri_node_mat[3,:,tri_counter] = cen_tri_node_mat[3,:,i,child2_index]
                tri_counter += 1
        else:
            if isin_fkt(child1_index, visited) == False:
                stack[child1_index] = child1_index
            if isin_fkt(child2_index, visited) == False:
                stack[child2_index] = child2_index

@njit
def find_N_tri_in_box(box_index, new_index_array, NTri_in_leafs_vec):
    array_len = int(len(new_index_array[0,:]))
    stack = np.ones(array_len+1)*(999999)
    visited = np.ones(array_len)*(-1)
    if new_index_array[0,box_index] == -9:
        return 1
    stack[0] = new_index_array[0,box_index]
    stack[1] = new_index_array[1,box_index]
    current_node = 0
    tot_tri = 0
    while current_node != 999999:
        current_node = int(np.min(stack))
        if current_node == 999999:
            return tot_tri
        stack_index = np.where(stack == current_node)[0][0]
        #print(current_node)
        visited[current_node] = current_node
        stack[stack_index] = 999999
        child1_index = (new_index_array[0,current_node])
        child1_index = int(child1_index)
        child2_index = (new_index_array[1,current_node])
        child2_index = int(child2_index)
        if child1_index == -9:
            tot_tri += NTri_in_leafs_vec[child2_index]
        else:
            if isin_fkt(child1_index, visited) == False:
                stack[child1_index] = child1_index
            if isin_fkt(child2_index, visited) == False:
                stack[child2_index] = child2_index

#print(find_N_tri_in_box(502, new_index_array, NTri_in_leafs_vec))


# from load_model_center_rotate import load_big_model
# from better_mesh_fkt import better_mesh_fkt
# from BVH_plot_bounding_volume_and_vertex import plot_BV_and_vertex
# Triangle_matrix, Triangle_normal_matrix = load_big_model('t90a_trekanter')

# Triangle_matrix = better_mesh_fkt(Triangle_matrix, 0.4)

# box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec = BVH_builder(Triangle_matrix, 300)
# index_Array, N_diff_vec = tree_node_fkt(box_node_mat, cen_tri_node_mat, N_box_vec, parent_vec)

# new_index_array = optimize_index_array(index_Array, box_node_mat, cen_tri_node_mat, N_diff_vec)

# NTri_in_leafs_vec = find_tri_in_leafs(cen_tri_node_mat)

# plot_BV_and_vertex(box_node_mat, 12, new_index_array, NTri_in_leafs_vec, cen_tri_node_mat, np.array([-3,3]), np.array([-7,4.5]), np.array([0,4.5]), 30, -30)

            