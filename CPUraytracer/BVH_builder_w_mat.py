# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 12:33:24 2024

@author: thoms
"""

'''
new BVH builder with material properties
'''

import numpy as np

from BVH_library_funcs import get_subset_fkt, is_splittable, tri_in_leaf
from BVH_library_funcs import calculate_centroid_fkt, first_bounding_box_fkt, split_triangles_w_material_fkt, create_bounding_box_fkt, max_tri_in_array, find_longest_axis

def BVH_builder_materials(Triangle_matrix, material_mat, max_in_leaf):
    primary_box = first_bounding_box_fkt(Triangle_matrix)
    box_node_mat = np.zeros((6,1))
    box_node_mat[:,0] = primary_box

    centroid_triangle_mat = calculate_centroid_fkt(Triangle_matrix)
    N_tri = len(centroid_triangle_mat[0,0,:])
    cen_tri_node_mat = np.zeros((4,3,N_tri,1))
    cen_tri_node_mat[:,:,:,0] =  centroid_triangle_mat
    
    material_mat_node = np.zeros((2,N_tri,1))
    material_mat_node[:,:,0] = material_mat
    
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
        material_mat_node_new = np.ones((2,max_N_tri,N_parents*2))*(-99)
        box_node_mat_new = np.zeros((6,N_parents*2 + N_box))
        box_node_mat_new[:,:(N_box)] = box_node_mat
        no_split_counter = 0
        N_box_vec[iteration_counter] = N_box
        for i in range(N_parents):
            subset_N = get_subset_fkt(cen_tri_node_mat[:,:,:,i])
            cen_tri_subset = cen_tri_node_mat[:,:,:subset_N,i]
            material_subset = material_mat_node[:,:subset_N,i]
            
            box_subset = box_node_mat[:,N_box - N_box_prev + i - 1]
            split_var = is_splittable(cen_tri_subset)
            if split_var == False:
                cen_tri_node_mat_new[:,:,0,i*2 - no_split_counter] = cen_tri_subset[:,:,0]
                material_mat_node_new[:,0,i*2 - no_split_counter] = material_subset[:,0]
                Box = create_bounding_box_fkt(cen_tri_subset)
                box_node_mat_new[:,N_box + i*2 - no_split_counter] = Box
                no_split_counter += 1
                global_no_split_vec[iteration_counter] += 1
                parent_vec[iteration_counter, i] = i
                continue
            
            axis_index, midpoint = find_longest_axis(box_subset)
           
            
            M1, M2, C1, C2, Mat1, Mat2 = split_triangles_w_material_fkt(cen_tri_subset, material_subset, axis_index)
                
            cen_tri_node_mat_new[:,:,:C1,i*2 - no_split_counter] = M1
            cen_tri_node_mat_new[:,:,:C2,(i+1)*2-1 - no_split_counter] = M2
            
            material_mat_node_new[:,:C1,i*2 - no_split_counter] = Mat1
            material_mat_node_new[:,:C2,(i+1)*2-1 - no_split_counter] = Mat1
            
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
        
        material_mat_node = material_mat_node_new[:,:,:(N_parents*2 - no_split_counter)]
        material_mat_node = material_mat_node[:,:max_tri_in_set,:]
        print(min_tri_in_leaf) 
        
    return box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec, material_mat_node