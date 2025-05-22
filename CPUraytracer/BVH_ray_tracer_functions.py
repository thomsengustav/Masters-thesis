# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 15:21:28 2024

@author: thoms
"""


'''
BVH ray tracing functions!
'''


#### ray tracer functions for BVH
from Hit_functions_fkt import hit_triangle_fkt, hit_sphere_fkt, hit_plane_fkt
from BVH import find_tree_ray_intersection_fkt
from numba import njit, prange
import numpy as np


@njit
def closest_object_primary_BVH_fkt(Triangle_matrix, Triangle_normal_matrix,
                                   new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec, # load geometry
                                   max_parameter_t, ray_origin, ray_direction):
    
    #print(ray_direction)
    t, child2_index, tri_index = find_tree_ray_intersection_fkt(ray_origin, ray_direction, new_index_array, box_node_mat, cen_tri_node_mat, max_parameter_t, NTri_in_leafs_vec)
    surface_norm = Triangle_normal_matrix[:, tri_index, child2_index]
    #print(t)
    
    if t == False:
        return False, surface_norm # major jank! Should just return False
    
    return t, surface_norm

from intersection_and_reflection_fkt import intercetion_point_fkt, plane_tri_ref_fkt, normal_direction_check_fkt
from primary_launch_fkt import check_normal_fkt, secondary_spread_fkt, secondary_spread_method2_fkt
from fast_vec_operators import len_fkt, dot_fkt

@njit(parallel=True)
def raster_intersection_fkt(N_primary, ray_origin_source, ray_direction_matrix, ray_spread, 
                             Triangle_matrix, Triangle_normal_matrix, 
                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                             max_parameter_t):
    distance_vec = np.zeros(N_primary)
    weight_vec = np.zeros(N_primary)
    for m in prange(N_primary):
        ray_origin = ray_origin_source
        ray_direction = ray_direction_matrix[m,:]
        parameter_object = closest_object_primary_BVH_fkt(Triangle_matrix, Triangle_normal_matrix, 
                               new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                               max_parameter_t, ray_origin, ray_direction)
        if parameter_object[0] == False:
            continue
        t_parameter = parameter_object[0]
        suface_normal = parameter_object[1]
        suface_normal = np.ascontiguousarray(suface_normal)
        intersection_point = intercetion_point_fkt(ray_origin, ray_direction, t_parameter)
        
        ray_surface_dot, suface_normal = normal_direction_check_fkt(ray_direction, suface_normal)
        distance_vec[m] = 2*len_fkt(intersection_point - ray_origin_source)
        weight_vec[m] = dot_fkt(ray_direction, -suface_normal)
        
    return distance_vec, weight_vec

@njit(parallel=True)
def raster_intersection_PW_fkt(N_primary, ray_origin_source, ray_direction_vec, ray_spread, 
                             Triangle_matrix, Triangle_normal_matrix, 
                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                             max_parameter_t):
    distance_vec = np.zeros(N_primary)
    weight_vec = np.zeros(N_primary)
    for m in prange(N_primary):
        ray_origin = ray_origin_source[m,:]
        ray_direction = ray_direction_vec
        parameter_object = closest_object_primary_BVH_fkt(Triangle_matrix, Triangle_normal_matrix, 
                               new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                               max_parameter_t, ray_origin, ray_direction)
        if parameter_object[0] == False:
            continue
        t_parameter = parameter_object[0]
        intersection_point = intercetion_point_fkt(ray_origin, ray_direction, t_parameter)
    
        distance_vec[m] = 2*np.linalg.norm(intersection_point - ray_origin)
        weight_vec[m] = 1
        
    return distance_vec, weight_vec

@njit(parallel=True)
def foot_print_intersection(N_primary, ray_origin_source, ray_direction_matrix, ray_spread, 
                             Triangle_matrix, Triangle_normal_matrix, 
                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                             max_parameter_t):
    int_points = np.zeros((3,N_primary))
    false_counter = 0
    for m in prange(N_primary):
        ray_origin = ray_origin_source
        ray_direction = ray_direction_matrix[m,:]
        parameter_object = closest_object_primary_BVH_fkt(Triangle_matrix, Triangle_normal_matrix, 
                               new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                               max_parameter_t, ray_origin, ray_direction)
        if parameter_object[0] == False:
            false_counter += 1 
            continue
        t_parameter = parameter_object[0]
        intersection_point = intercetion_point_fkt(ray_origin, ray_direction, t_parameter)
        int_points[:,m-false_counter] = intersection_point
    
    ## remove misses todo
    int_points_n = int_points[:,:(N_primary-false_counter)]
    return int_points_n

@njit(parallel=True)
def foot_print_planewave_intersection(N_primary, ray_origin_source_mat, ray_direction_var, ray_spread, 
                             Triangle_matrix, Triangle_normal_matrix, 
                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                             max_parameter_t):
    int_points = np.zeros((3,N_primary))
    false_counter = 0
    for m in prange(N_primary):
        ray_origin = ray_origin_source_mat[m,:]
        ray_direction = ray_direction_var
        parameter_object = closest_object_primary_BVH_fkt(Triangle_matrix, Triangle_normal_matrix, 
                               new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                               max_parameter_t, ray_origin, ray_direction)
        if parameter_object[0] == False:
            false_counter += 1 
            continue
        t_parameter = parameter_object[0]
        intersection_point = intercetion_point_fkt(ray_origin, ray_direction, t_parameter)
        int_points[:,m-false_counter] = intersection_point
    
    ## remove misses todo
    int_points_n = int_points[:,:(N_primary-false_counter)]
    return int_points_n



@njit(parallel=True)
def Primary_ray_iteration_BVH_fkt(N_primary, N_multiply, ray_origin_source, ray_direction_matrix, 
                             ray_spread, 
                             Triangle_matrix, Triangle_normal_matrix, 
                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                             max_parameter_t):
    
    N_secondary = N_primary * N_multiply
    distance_vec = np.zeros(N_secondary)
    intersection_point_mat = np.zeros((N_secondary,3))
    reflection_vec_mat = np.zeros((N_secondary,3))
    for m in prange(N_primary):
        ray_origin = ray_origin_source
        ray_direction = ray_direction_matrix[m,:]
        parameter_object = closest_object_primary_BVH_fkt(Triangle_matrix, Triangle_normal_matrix, 
                               new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                               max_parameter_t, ray_origin, ray_direction)
        if parameter_object[0] == False:
            continue
        t_parameter = parameter_object[0]
        suface_normal = parameter_object[1]
        suface_normal = np.ascontiguousarray(suface_normal)
        intersection_point = intercetion_point_fkt(ray_origin, ray_direction, t_parameter)
        reflection_vec = plane_tri_ref_fkt(ray_direction, suface_normal)
        
        suface_normal = check_normal_fkt(suface_normal, reflection_vec)
        secondary_direction_mat = secondary_spread_fkt(intersection_point, reflection_vec, ray_spread, N_multiply, suface_normal)
        # secondary_direction_mat = secondary_spread_method2_fkt(intersection_point, reflection_vec, ray_spread, N_multiply, suface_normal)
        reflection_vec_mat[m*N_multiply:(m+1)*N_multiply, :] = secondary_direction_mat
        intersection_point_mat[m*N_multiply:(m+1)*N_multiply, :] = intersection_point
        distance_vec[m*N_multiply:(m+1)*N_multiply] = t_parameter
    
    return intersection_point_mat, reflection_vec_mat, distance_vec

@njit
def closest_object_BVH_fkt(Triangle_matrix, Triangle_normal_matrix, 
                       new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,
                       det_center, det_radius, # load geometry
                       max_parameter_t, ray_origin, ray_direction):
    
    t, child2_index, tri_index = find_tree_ray_intersection_fkt(ray_origin, ray_direction, new_index_array, box_node_mat, cen_tri_node_mat, max_parameter_t, NTri_in_leafs_vec,)
    surface_norm = Triangle_normal_matrix[:, tri_index, child2_index]
    #print(f"t {t}")
    hit_det = hit_sphere_fkt(ray_origin, ray_direction, det_center, det_radius)
    #print(f"det_t {hit_det}")
    if hit_det < t and hit_det > 0:
        t = hit_det
            
    if t == max_parameter_t:
        return False, surface_norm
    else:
        return t, surface_norm

from closest_intersection_fkt import detector_check_fkt, detector_lenght_cal_fkt

@njit(parallel=True)
def main_ray_iteration_BVH_fkt(N_bounces, N_rays, max_parameter_t, ray_origin_matrix, ray_direction_matrix, 
                             det_center, det_radius, distance_vec, 
                             Triangle_matrix, Triangle_normal_matrix, 
                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec):
    distance_vec_final = np.zeros(N_rays)
    for m in prange(N_rays):
        ray_origin = ray_origin_matrix[m,:]
        ray_direction = ray_direction_matrix[m,:]
        last_hit = ray_origin
        distance = distance_vec[m]
        for n in range(N_bounces):
            parameter_object = closest_object_BVH_fkt(Triangle_matrix, Triangle_normal_matrix, 
                                   new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,
                                   det_center, det_radius, # load geometry
                                   max_parameter_t, ray_origin, ray_direction)
            t_parameter = parameter_object[0]
            hit_det = hit_sphere_fkt(ray_origin, ray_direction, det_center, det_radius)
            
            if t_parameter == False:
                if hit_det == False:
                    break
                else:
                    distance_vec_final[m] = detector_lenght_cal_fkt(det_center, last_hit) + distance
                    break
            
            suface_normal = parameter_object[1]
            ray_origin = intercetion_point_fkt(ray_origin, ray_direction, t_parameter)
            distance = distance + t_parameter
            ray_direction = plane_tri_ref_fkt(ray_direction, suface_normal)
            last_hit = ray_origin
            
    return distance_vec_final



@njit(parallel=True)
def main_ray_iteration_BVH_test_fkt(N_bounces, N_rays, max_parameter_t, ray_origin_matrix, ray_direction_matrix, 
                             det_center, det_radius, distance_vec, 
                             Triangle_matrix, Triangle_normal_matrix, 
                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec):
    distance_vec_final = np.zeros(N_rays)
    plot_rays_mat = np.zeros((3,N_rays,N_bounces))
    for m in prange(N_rays):
        ray_origin = ray_origin_matrix[m,:]
        ray_direction = ray_direction_matrix[m,:]
        last_hit = ray_origin
        distance = distance_vec[m]
        for n in range(N_bounces):
            parameter_object = closest_object_BVH_fkt(Triangle_matrix, Triangle_normal_matrix, 
                                   new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,
                                   det_center, det_radius, # load geometry
                                   max_parameter_t, ray_origin, ray_direction)
            t_parameter = parameter_object[0]
            hit_det = hit_sphere_fkt(ray_origin, ray_direction, det_center, det_radius)
            
            if t_parameter == False:
                if hit_det == False:
                    break
                else:
                    distance_vec_final[m] = detector_lenght_cal_fkt(det_center, last_hit) + distance
                    break
            
            suface_normal = parameter_object[1]
            ray_origin = intercetion_point_fkt(ray_origin, ray_direction, t_parameter)
            plot_rays_mat[:,m,n] = ray_origin
            distance = distance + t_parameter
            ray_direction = plane_tri_ref_fkt(ray_direction, suface_normal)
            last_hit = ray_origin
            
    return distance_vec_final, plot_rays_mat
