# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 10:01:34 2025

@author: thoms
"""

'''
Run all 4 polarizations off one ray tracing sim.
generate two E-fields pr. ray and at detector
'''
import numpy as np
from numba import njit, prange

from Hit_functions_fkt import hit_sphere_fkt
from intersection_and_reflection_fkt import plane_tri_ref_fkt
from RT_polarization import get_H_normal_vec, get_V_normal_vec_fin, get_transformation_mat
from rotation_matrix_fkt import beam_direction_fkt, rotation_matrix_fkt
from ray_generation_function import launch_ray_components_uniform_fkt
from occlusion_test import occlusion_test_fkt
from RT_polarization import H_lin_pol_filter, V_lin_pol_filter, fres_reflection_HV
from fast_vec_operators import norm_vec_fkt, len_fkt, mat_vec_prod
from BVH import find_tree_ray_intersection_fkt

@njit
def detector_lenght_cal_fkt(det_center, last_hit):
    new_vec = last_hit - det_center
    distance = np.sqrt(new_vec[0]**2 + new_vec[1]**2 + new_vec[2]**2)
    return distance

# generate ray_direction_matrix, ray_origin_matrix, E_field_matrix [virker!]
@njit
def get_HV_field_source(N_rays, Radar_location, beam_size): 
    ray_direction_matrix = np.zeros((N_rays, 3))
    ray_origin_matrix = np.zeros((N_rays, 3))
    ray_H_field_matrix = np.zeros((N_rays, 3))
    ray_V_field_matrix = np.zeros((N_rays, 3))
    trans_matrix = np.zeros((3,3,N_rays))
    
    ray_origin_matrix[:,:] = Radar_location
    
    beam_direction = beam_direction_fkt(Radar_location)
    rotation_matrix = rotation_matrix_fkt(beam_direction)
    ray_direction_matrix[:,0], ray_direction_matrix[:,1], ray_direction_matrix[:,2] = launch_ray_components_uniform_fkt(N_rays, beam_size, rotation_matrix, Radar_location)
    
    H_field_VHK = np.array([0,1.0,0]) # H pol at source
    V_field_VHK = np.array([1.0,0,0]) # V pol at source
    
    for i in range(N_rays):
        ray_dir = ray_direction_matrix[i,:]
        V_norm = get_V_normal_vec_fin(ray_dir)
        H_norm = get_H_normal_vec(ray_dir, V_norm)
        
        trans_mat = get_transformation_mat(V_norm, H_norm, ray_dir)
        H_field_xyz = mat_vec_prod(trans_mat, H_field_VHK)
        ray_H_field_matrix[i,:] = H_field_xyz
        V_field_xyz = mat_vec_prod(trans_mat, V_field_VHK)
        ray_V_field_matrix[i,:] = V_field_xyz
        
        trans_matrix[:,:,i] = trans_mat
    return ray_origin_matrix, ray_direction_matrix, trans_matrix, ray_H_field_matrix, ray_V_field_matrix

@njit
def priamry_ray_surface_int_HV(ray_origin, ray_direction, H_vec_xyz, V_vec_xyz, distance,
                            Triangle_normal_matrix, Material_mat, cen_tri_node_mat,
                            t_parameter, tri_index, child2_index):
    suface_normal = Triangle_normal_matrix[:, tri_index, child2_index]
    mat_vec = Material_mat[:,tri_index, child2_index] 
    centroid = cen_tri_node_mat[3,:,tri_index, child2_index]
    
    int_point = ray_origin + ray_direction*t_parameter
    ray_direction_out = plane_tri_ref_fkt(ray_direction, suface_normal)
    distance = len_fkt(centroid - ray_origin) + distance
    
    n_index = mat_vec[0]
    reflectivity = mat_vec[1] # random weight variable from normal distribution
    
    H_prev_xyz = H_vec_xyz
    V_prev_xyz = V_vec_xyz
    H_vec_xyz, V_vec_xyz = fres_reflection_HV(H_vec_xyz, V_vec_xyz, 1, n_index, ray_direction, ray_direction_out, suface_normal)

    return H_prev_xyz, V_prev_xyz, H_vec_xyz, V_vec_xyz, distance, ray_direction_out, int_point, centroid, reflectivity

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
def get_BS_weight_HV(H_out, V_out, trans_mat):
    # through detector polariser
    trans_inv = np.linalg.inv(trans_mat)
    H_vec_VHK = mat_vec_prod(trans_inv, H_out)
    V_vec_VHK = mat_vec_prod(trans_inv, V_out)
    
    HH_final = H_lin_pol_filter(H_vec_VHK)
    HV_final = V_lin_pol_filter(H_vec_VHK)
    VH_final = H_lin_pol_filter(V_vec_VHK)
    VV_final = V_lin_pol_filter(V_vec_VHK)
    # size of E_vec
    HH_int = len_fkt(HH_final)
    HV_int = len_fkt(HV_final)
    VH_int = len_fkt(VH_final)
    VV_int = len_fkt(VV_final)
    return HH_int, HV_int, VH_int, VV_int

@njit
def rotate_BS_E_field(H_vec_xyz, V_vec_xyz, ray_direction_out):
    rot_mat = rotation_matrix_fkt(ray_direction_out)
    H_BSrot_xyz = np.zeros(3)
    V_BSrot_xyz = np.zeros(3)
    
    H_BSrot_xyz[0] = H_vec_xyz[0] * rot_mat[0,0] + H_vec_xyz[1] * rot_mat[0,1] + H_vec_xyz[2] * rot_mat[0,2]
    H_BSrot_xyz[1] = H_vec_xyz[0] * rot_mat[1,0] + H_vec_xyz[1] * rot_mat[1,1] + H_vec_xyz[2] * rot_mat[1,2]
    H_BSrot_xyz[2] = H_vec_xyz[0] * rot_mat[2,0] + H_vec_xyz[1] * rot_mat[2,1] + H_vec_xyz[2] * rot_mat[2,2]
    
    V_BSrot_xyz[0] = V_vec_xyz[0] * rot_mat[0,0] + V_vec_xyz[1] * rot_mat[0,1] + V_vec_xyz[2] * rot_mat[0,2]
    V_BSrot_xyz[1] = V_vec_xyz[0] * rot_mat[1,0] + V_vec_xyz[1] * rot_mat[1,1] + V_vec_xyz[2] * rot_mat[1,2]
    V_BSrot_xyz[2] = V_vec_xyz[0] * rot_mat[2,0] + V_vec_xyz[1] * rot_mat[2,1] + V_vec_xyz[2] * rot_mat[2,2]
    
    return H_BSrot_xyz, V_BSrot_xyz


@njit(parallel=True)
def main_ray_BVH_Pol_HV_fkt(N_bounces, N_rays, max_parameter_t, ray_origin_matrix, ray_direction_matrix, 
                             det_center, det_radius, 
                             Triangle_normal_matrix, Material_mat,
                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,
                             H_source_mat, V_source_mat, trans_martrix, ref_size):
    
    N_bounces_1 = N_bounces + 1
    ## include BC in distance and include weight!
    distance_vec = np.zeros(N_rays*(N_bounces_1))
    Weight_HH = np.zeros(N_rays*(N_bounces_1))
    Weight_HV = np.zeros(N_rays*(N_bounces_1))
    Weight_VH = np.zeros(N_rays*(N_bounces_1))
    Weight_VV = np.zeros(N_rays*(N_bounces_1))
    for m in prange(N_rays):
        ray_origin = ray_origin_matrix[m,:]
        ray_direction = ray_direction_matrix[m,:]
        
        ### first hit calculation
        t_parameter, child2_index, tri_index = find_tree_ray_intersection_fkt(ray_origin, ray_direction, new_index_array, box_node_mat, cen_tri_node_mat, max_parameter_t, NTri_in_leafs_vec)
        
        if t_parameter == False:
            continue
        
        trans_mat = trans_martrix[:,:,m]
        H_vec_xyz = H_source_mat[m,:]
        H_prev_xyz = np.array([0,0,0.0])
        V_vec_xyz = V_source_mat[m,:]
        V_prev_xyz = np.array([0,0,0.0])
        distance = 0
        
        H_prev_xyz, V_prev_xyz, H_vec_xyz, V_vec_xyz, distance, ray_direction_out, int_point, centroid, reflectivity = priamry_ray_surface_int_HV(ray_origin, ray_direction, H_vec_xyz, V_vec_xyz, distance,
                                                                                                              Triangle_normal_matrix, Material_mat, cen_tri_node_mat,
                                                                                                             t_parameter, tri_index, child2_index)
        # rotate E-fields toward detector!
        H_BSrot_xyz, V_BSrot_xyz = rotate_BS_E_field(H_vec_xyz, V_vec_xyz, ray_direction_out)
        dot_prod = np.dot(ray_direction_out, ray_direction)
        
        # BS
        det_vector = det_center - centroid
        d_int_det = len_fkt(det_vector)
        HH_int, HV_int, VH_int, VV_int = get_BS_weight_HV(H_BSrot_xyz, V_BSrot_xyz, trans_mat)
        distance_vec[m] = distance + d_int_det
        Weight_HH[m] = HH_int * reflectivity# * dot_prod
        Weight_HV[m] = HV_int * reflectivity# * dot_prod
        Weight_VH[m] = VH_int * reflectivity# * dot_prod
        Weight_VV[m] = VV_int * reflectivity# * dot_prod
        
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
                    HH_int, HV_int, VH_int, VV_int = get_BS_weight_HV(H_vec_xyz, V_vec_xyz, trans_mat)
                    Weight_HH[m + n*N_rays] = HH_int * ref_size
                    Weight_HV[m + n*N_rays] = HV_int * ref_size
                    Weight_VH[m + n*N_rays] = VH_int * ref_size
                    Weight_VV[m + n*N_rays] = VV_int * ref_size
                    break
            
            H_prev_xyz, V_prev_xyz, H_vec_xyz, V_vec_xyz, distance, ray_direction_out, int_point, centroid, reflectivity = priamry_ray_surface_int_HV(ray_origin, ray_direction, H_vec_xyz, V_vec_xyz, distance,
                                                                                                                  Triangle_normal_matrix, Material_mat, cen_tri_node_mat,
                                                                                                                 t_parameter, tri_index, child2_index)
            
            # BS
            d_int_det =  BS_fkt(det_center, int_point, centroid, new_index_array, box_node_mat, cen_tri_node_mat, max_parameter_t, NTri_in_leafs_vec)
            if d_int_det != False:
                distance_vec[m + n*N_rays] = distance + d_int_det
                H_BSrot_xyz, V_BSrot_xyz = rotate_BS_E_field(H_vec_xyz, V_vec_xyz, ray_direction_out)
                HH_int, HV_int, VH_int, VV_int = get_BS_weight_HV(H_BSrot_xyz, V_BSrot_xyz, trans_mat)# * np.dot(ray_direction_out, ray_direction)
                Weight_HH[m + n*N_rays] = HH_int * reflectivity
                Weight_HV[m + n*N_rays] = HV_int * reflectivity
                Weight_VH[m + n*N_rays] = VH_int * reflectivity
                Weight_VV[m + n*N_rays] = VV_int * reflectivity
            
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
                        HH_int, HV_int, VH_int, VV_int = get_BS_weight_HV(H_vec_xyz, V_vec_xyz, trans_mat)
                        Weight_HH[m + N_bounces*N_rays] = HH_int * ref_size
                        Weight_HV[m + N_bounces*N_rays] = HV_int * ref_size
                        Weight_VH[m + N_bounces*N_rays] = VH_int * ref_size
                        Weight_VV[m + N_bounces*N_rays] = VV_int * ref_size
                        break
        
    return distance_vec, Weight_HH, Weight_HV, Weight_VH, Weight_VV

from remove_missed_rays_HV import remove_missed_rays_HV

# BVH_data = (Triangle_normal_matrix, material_mat_node, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)
# SETUP_data = (radar_range, slant_angle, azimuth_start, azimuth_end, azimuth_steps, max_parameter_t, max_in_leaf, max_dist,
#               n_index, N_rays, N_bounces, Detector_radius, beam_size, ref_size, signal_resolution)

@njit
def main_loop_HV_fkt(N_rays, N_bounces, beam_size, Radar_location, max_parameter_t, ref_size, Detector_radius, BVH_data):
    ray_origin_matrix, ray_direction_matrix, trans_matrix, ray_H_field_matrix, ray_V_field_matrix = get_HV_field_source(N_rays, Radar_location, beam_size)
    
    distance_vec, Weight_HH, Weight_HV, Weight_VH, Weight_VV = main_ray_BVH_Pol_HV_fkt(N_bounces, N_rays, max_parameter_t, ray_origin_matrix, ray_direction_matrix, 
                                                    Radar_location, Detector_radius, 
                                                    BVH_data[0], BVH_data[1],
                                                    BVH_data[2], BVH_data[3], BVH_data[4], BVH_data[5],
                                                    ray_H_field_matrix, ray_V_field_matrix, trans_matrix, ref_size)
    new_distance_vec, new_weight_HH, new_weight_HV, new_weight_VH, new_weight_VV = remove_missed_rays_HV(distance_vec, Weight_HH, Weight_HV, Weight_VH, Weight_VV)
    return new_distance_vec, new_weight_HH, new_weight_HV, new_weight_VH, new_weight_VV

from range_histogram_data import get_range_histogram_HV, remove_zeros_range_HV
from setup_azimuthal_arc import azimuth_radar_location_fkt, save_distance_fkt
from loading_and_est_time import loading_bar, get_time_est
import time


def SAR_pol_HV(pol_name, SETUP_data, BVH_data):
    radar_range, slant_angle = SETUP_data[0], SETUP_data[1]
    azimuth_start, azimuth_end, azimuth_steps = SETUP_data[2], SETUP_data[3], SETUP_data[4]
    N_rays, N_bounces, max_parameter_t = SETUP_data[9], SETUP_data[10], SETUP_data[5]
    beam_size, ref_size, signal_res = SETUP_data[12], SETUP_data[13], SETUP_data[14]
    Detector_radius = SETUP_data[11]
    
    radar_loc_mat = azimuth_radar_location_fkt(slant_angle, radar_range,
                                   azimuth_steps, azimuth_start, azimuth_end)
    name_counter = 0
    tot_time = 0
    for azi in range(azimuth_steps):
        name_counter += 1
        time_1 = time.time()
        Rast_name_azi = "HVRT_azi_" + str(name_counter) + "_start_" + str(azimuth_start) + "_end_" + str(azimuth_end) + "_" + pol_name
        Radar_location = radar_loc_mat[azi,:]
        
        distance_vec, HH, HV, VH, VV = main_loop_HV_fkt(N_rays, N_bounces, beam_size, Radar_location, max_parameter_t, ref_size, Detector_radius, BVH_data)
        
        range_vec, his_HH, his_HV, his_VH, his_VV = get_range_histogram_HV(distance_vec, HH, HV, VH, VV, max_parameter_t, signal_res)
        
        fin_range_vec, fin_HH, fin_HV, fin_VH, fin_VV = remove_zeros_range_HV(range_vec, his_HH, his_HV, his_VH, his_VV)
        
        Len_data = int(len(fin_range_vec))
        save_data = np.zeros((Len_data,5))
        save_data[:,0], save_data[:,1], save_data[:,2], save_data[:,3], save_data[:,4] = fin_range_vec, fin_HH, fin_HV, fin_VH, fin_VV
        save_distance_fkt(save_data, Rast_name_azi)
        
        loading_bar(name_counter, azimuth_steps)
        tot_time = get_time_est(time_1, tot_time, azimuth_steps, name_counter)

