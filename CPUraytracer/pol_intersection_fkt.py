# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 21:30:11 2024

@author: thoms
"""

'''
ray tracing with clutter and backscattering! Main closeset hit function!

Functions for one polarization at the time. 
'''

from BVH import find_tree_ray_intersection_fkt
from numba import njit, prange
import numpy as np
from fast_vec_operators import norm_vec_fkt, len_fkt, mat_vec_prod

@njit
def detector_lenght_cal_fkt(det_center, last_hit):
    new_vec = last_hit - det_center
    distance = np.sqrt(new_vec[0]**2 + new_vec[1]**2 + new_vec[2]**2)
    return distance

@njit
def closest_object_w_material_BVH_fkt(new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec, # load geometry
                                   max_parameter_t, ray_origin, ray_direction):
    
    t, child2_index, tri_index = find_tree_ray_intersection_fkt(ray_origin, ray_direction, new_index_array, box_node_mat, cen_tri_node_mat, max_parameter_t, NTri_in_leafs_vec)
    # surface_norm = Triangle_normal_matrix[:, tri_index, child2_index]
    # mat_vec = Material_mat[:,tri_index, child2_index] # mat_vec contains two complex numbers, N = n+ik and ref = Rr + iRi  
    
    return t, child2_index, tri_index

'''
Need to impliment complex n, and R for each triangle in the scene and 
transform the matrix the same way as the BVH does, for compatible search with tri_index, child2_index 
from find_tree_ray_intersection().
'''

from Hit_functions_fkt import hit_sphere_fkt
from intersection_and_reflection_fkt import plane_tri_ref_fkt
from RT_polarization import fres_reflection_fkt2, get_H_normal_vec, get_V_normal_vec_fin, get_transformation_mat
from rotation_matrix_fkt import beam_direction_fkt, rotation_matrix_fkt
from ray_generation_function import launch_ray_components_uniform_fkt
from occlusion_test import occlusion_test_fkt
from RT_polarization import H_lin_pol_filter, V_lin_pol_filter


# generate ray_direction_matrix, ray_origin_matrix, E_field_matrix [virker!]
@njit
def get_E_field_source(N_rays, pol_orientation, Radar_location, beam_size): 
    ray_direction_matrix = np.zeros((N_rays, 3))
    ray_origin_matrix = np.zeros((N_rays, 3))
    ray_E_field_matrix = np.zeros((N_rays, 3))
    trans_matrix = np.zeros((3,3,N_rays))
    
    ray_origin_matrix[:,:] = Radar_location
    
    beam_direction = beam_direction_fkt(Radar_location)
    rotation_matrix = rotation_matrix_fkt(beam_direction)
    ray_direction_matrix[:,0], ray_direction_matrix[:,1], ray_direction_matrix[:,2] = launch_ray_components_uniform_fkt(N_rays, beam_size, rotation_matrix, Radar_location)
    
    if pol_orientation[0] == 'H':
        E_field_VHK = np.array([0,1.0,0]) # H pol at source
    else:
        E_field_VHK = np.array([1.0,0,0]) # V pol at source
    
    for i in range(N_rays):
        ray_dir = ray_direction_matrix[i,:]
        V_norm = get_V_normal_vec_fin(ray_dir)
        H_norm = get_H_normal_vec(ray_dir, V_norm)
        
        trans_mat = get_transformation_mat(V_norm, H_norm, ray_dir)
        E_field_xyz = mat_vec_prod(trans_mat, E_field_VHK)
        ray_E_field_matrix[i,:] = E_field_xyz
        trans_matrix[:,:,i] = trans_mat
    return ray_origin_matrix, ray_direction_matrix, ray_E_field_matrix, trans_matrix

# gets distance from ray-surface intersection to detector and checks for occlusion
@njit
def BS_fkt(det_center, intersection_point, centroid, index_Array, box_node_mat, cen_tri_node_mat, t_max, NTri_in_leafs_vec):
    # calculate vec to detector
    det_vector = det_center - intersection_point
    det_vector_norm = norm_vec_fkt(det_vector)
    # check for occlusion
    occlusion_bool = occlusion_test_fkt(intersection_point, det_vector_norm, index_Array, box_node_mat, cen_tri_node_mat, t_max, NTri_in_leafs_vec)
    if occlusion_bool == False:
        return False
    else:
        d_int_det = len_fkt(det_center - centroid)
        return d_int_det 

@njit
def get_BS_weight(E_in, E_out, trans_mat, det_pol):
    E_avg_xyz = E_out# (E_in + E_out) / 2 # mix E_vec of incomming and outgoing. Just a test
    # through detector polariser
    trans_inv = np.linalg.inv(trans_mat)
    E_vec_VHK = mat_vec_prod(trans_inv, E_avg_xyz)
    if det_pol == False:
        E_final = H_lin_pol_filter(E_vec_VHK)
    else:
        E_final = V_lin_pol_filter(E_vec_VHK)
    E_int = len_fkt(E_final) # size of E_vec
    return E_int

@njit
def priamry_ray_surface_int(ray_origin, ray_direction, E_vec_xyz, distance,
                            Triangle_normal_matrix, Material_mat, cen_tri_node_mat,
                            t_parameter, tri_index, child2_index):
    suface_normal = Triangle_normal_matrix[:, tri_index, child2_index]
    mat_vec = Material_mat[:,tri_index, child2_index] 
    centroid = cen_tri_node_mat[3,:,tri_index, child2_index]
    
    int_point = ray_origin + ray_direction*t_parameter
    ray_direction_out = plane_tri_ref_fkt(ray_direction, suface_normal)
    distance = len_fkt(centroid - ray_origin) + distance
    
    n_index = mat_vec[0]
    reflectivity = mat_vec[1] # random weight varianble form normal distribution
    
    E_prev_xyz = E_vec_xyz
    E_vec_xyz = fres_reflection_fkt2(E_vec_xyz, 1, n_index, ray_direction, ray_direction_out, suface_normal)

    return E_prev_xyz, E_vec_xyz, distance, ray_direction_out, int_point, centroid, reflectivity
    
@njit
def get_det_bool(pol_orientation):
    if pol_orientation[1] == 'H':
        det_pol = False
    elif pol_orientation[1] == 'V':
        det_pol = True
    else:
        print('pol_orientation error')
    return det_pol


@njit(parallel=True)
def main_ray_BVH_Pol_BS_fkt(N_bounces, N_rays, max_parameter_t, ray_origin_matrix, ray_direction_matrix, 
                             det_center, det_radius, 
                             Triangle_normal_matrix, Material_mat,
                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,
                             E_source_mat, trans_martrix, pol_orientation):
    
    det_pol = get_det_bool(pol_orientation)
    N_bounces_1 = N_bounces + 1
    ## include BC in distance and include weight!
    distance_vec = np.zeros(N_rays*(N_bounces_1))
    Weight_vec = np.zeros(N_rays*(N_bounces_1))
    for m in prange(N_rays):
        ray_origin = ray_origin_matrix[m,:]
        ray_direction = ray_direction_matrix[m,:]
        trans_mat = trans_martrix[:,:,m]
        E_vec_xyz = E_source_mat[m,:]
        E_prev_xyz = np.array([0,0,0.0])
        distance = 0
        ### first hit calculation
        t_parameter, child2_index, tri_index = find_tree_ray_intersection_fkt(ray_origin, ray_direction, new_index_array, box_node_mat, cen_tri_node_mat, max_parameter_t, NTri_in_leafs_vec)
        
        if t_parameter == False:
            continue
        
        E_prev_xyz, E_vec_xyz, distance, ray_direction_out, int_point, centroid, reflectivity = priamry_ray_surface_int(ray_origin, ray_direction, E_vec_xyz, distance,
                                                                                                              Triangle_normal_matrix, Material_mat, cen_tri_node_mat,
                                                                                                             t_parameter, tri_index, child2_index)
        
        # BS
        det_vector = det_center - centroid
        d_int_det = len_fkt(det_vector)
        E_int = get_BS_weight(E_prev_xyz, E_vec_xyz, trans_mat, det_pol)
        distance_vec[m] = distance + d_int_det
        Weight_vec[m] = E_int * reflectivity
        
        ray_origin = int_point
        ray_direction = ray_direction_out
        ## bounces hits
        for n in range(1,N_bounces):
            t_parameter, child2_index, tri_index = find_tree_ray_intersection_fkt(ray_origin, ray_direction, new_index_array, box_node_mat, cen_tri_node_mat, max_parameter_t, NTri_in_leafs_vec)
            hit_det = hit_sphere_fkt(ray_origin, ray_direction, det_center, det_radius)
            
            if t_parameter == False:
                if hit_det == False:
                    break
                else:
                    distance_vec[m + n*N_rays] = detector_lenght_cal_fkt(det_center, ray_origin) + distance
                    ## detect E-field 
                    trans_inv = np.linalg.inv(trans_mat)
                    E_vec_VHK = mat_vec_prod(trans_inv, E_vec_xyz) # trans_inv @ E_vec_xyz
                    if det_pol == False:
                        E_final = H_lin_pol_filter(E_vec_VHK)
                    else:
                        E_final = V_lin_pol_filter(E_vec_VHK)
                    Weight_vec[m + n*N_rays] = len_fkt(E_final) * 5 # *10 to weigh direct reflections more strongly
                    break
            
            E_prev_xyz, E_vec_xyz, distance, ray_direction_out, int_point, centroid, reflectivity = priamry_ray_surface_int(ray_origin, ray_direction, E_vec_xyz, distance,
                                                                                                                  Triangle_normal_matrix, Material_mat, cen_tri_node_mat,
                                                                                                                  t_parameter, tri_index, child2_index)
            
            # BS
            d_int_det =  BS_fkt(det_center, int_point, centroid, new_index_array, box_node_mat, cen_tri_node_mat, max_parameter_t, NTri_in_leafs_vec)
            if d_int_det != False:
                E_int = get_BS_weight(E_prev_xyz, E_vec_xyz, trans_mat, det_pol)
                distance_vec[m + n*N_rays] = distance + d_int_det
                Weight_vec[m + n*N_rays] = E_int * reflectivity
            
            ray_origin = int_point
            ray_direction = ray_direction_out
            
            if n == N_bounces:
                # final check!
                t_parameter, child2_index, tri_index = find_tree_ray_intersection_fkt(ray_origin, ray_direction, new_index_array, box_node_mat, cen_tri_node_mat, max_parameter_t, NTri_in_leafs_vec)
                hit_det = hit_sphere_fkt(ray_origin, ray_direction, det_center, det_radius)
                
                if t_parameter == False:
                    if hit_det == False:
                        break
                    else:
                        distance_vec[m + N_bounces*N_rays] = detector_lenght_cal_fkt(det_center, ray_origin) + distance
                        ## detect E-field 
                        trans_inv = np.linalg.inv(trans_mat)
                        E_vec_VHK = mat_vec_prod(trans_inv, E_vec_xyz) # trans_inv @ E_vec_xyz
                        if det_pol == False:
                            E_final = H_lin_pol_filter(E_vec_VHK)
                        else:
                            E_final = V_lin_pol_filter(E_vec_VHK)
                        Weight_vec[m + N_bounces*N_rays] = len_fkt(E_final) * 10 # *10 to weigh direct reflections more strongly
                        break
        
    return distance_vec, Weight_vec

