# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:13:57 2024

@author: thoms
"""

'''
shooting and bouncing ray polarization RT!
for one polarization at the time!
'''

# import packages
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
from fast_vec_operators import len_fkt


#### Initialize  parameters!

pol_name = "all_råsted_test_1" # name for saving data

### polarization parameters

pol_orientation = 'HH' # set orientation of emitter and detector (HH, VV, HV)

###
radar_range = 100 # meters - 0
slant_angle = 25 # degrees - 1

azimuth_start = -5 # degrees - 2
azimuth_end = 5 # degrees - 3
azimuth_steps = 120 # - 4

max_parameter_t = 150 # longest allowed ray in meters - 5
max_in_leaf = 200 # - 6
max_dist = 0.2 # - 7

n_index = 300 # - 8

N_rays = 500000 # number of primary rays launched from source # - 9
N_bounces = 3 # number of allowed bounces # - 10
Detector_radius = 15 # meters # - 11
beam_size = 4 # - 12
ref_size = 10 # - 13
signal_resolution = 0.001 # 1mm # - 14


SETUP_data = (radar_range, slant_angle, azimuth_start, azimuth_end, azimuth_steps, max_parameter_t, max_in_leaf, max_dist,
              n_index, N_rays, N_bounces, Detector_radius, beam_size, ref_size, signal_resolution)

include_BS = False         # controls inclusion of backscatter
include_clutter = False    # controls inclusion of clutter
get_4_pols = True # get HH, HV, VV, VH
get_3_pols = False # get HH, HV, VV

clutter_data = False
if include_clutter == True:
    clutter_center = np.array([0,0,0])
    clutter_area_L = 10
    dstep = 0.1
    
    from clutter_P_values import get_P_vec_tree
    P_vec = get_P_vec_tree(pol_orientation)
    
    clutter_data = (dstep, slant_angle, P_vec)

if include_BS == True:
    from BS_pol import BS_detection_weight
    BS_strength = 0.5 # fraction of incident e-field magnitude returned as unplarized light
    
    

#### import geometry
from load_gmesh_square import load_gmesh_triangle_model
from better_mesh_fkt import mesh_setup_fkt
from make_material_matrix import make_material_mat
from rotate_mesh_fkt import rotate_mesh
from load_model_center_rotate import load_big_model

# Triangle_matrix, Triangle_normal_matrix = load_gmesh_triangle_model('m_3_box.txt')
Triangle_matrix, Triangle_normal_matrix = load_big_model('råsted_vej3') 
Triangle_matrix = Triangle_matrix*200 # råsted_vej

def rotate_ford(Triangle_matrix):
    N = int(len(Triangle_matrix[0,0,:]))
    new_Triangle_matrix = np.zeros((3,3,N))
    new_Triangle_matrix[:,0,:] = Triangle_matrix[:,0,:]
    new_Triangle_matrix[:,1,:] = Triangle_matrix[:,2,:]
    new_Triangle_matrix[:,2,:] = Triangle_matrix[:,1,:]
    return new_Triangle_matrix

Triangle_matrix = rotate_ford(Triangle_matrix)

Triangle_matrix, Triangle_normal_matrix = mesh_setup_fkt(Triangle_matrix, max_dist)
Triangle_matrix = rotate_mesh(Triangle_matrix, 20)

Material_matrix = make_material_mat(Triangle_matrix, n_index, clutter_data)


from BVH import plot_bounding_volume, find_tri_in_leafs, tri_normal_calculator_BVH_fkt, tree_node_fkt, optimize_index_array
from centroid_on_rasterization import BVH_builder_materials
### setup BVH!
box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec, material_mat_node = BVH_builder_materials(Triangle_matrix, Material_matrix, max_in_leaf)

index_Array, N_diff_vec = tree_node_fkt(box_node_mat, cen_tri_node_mat, N_box_vec, parent_vec)
new_index_array = optimize_index_array(index_Array, box_node_mat, cen_tri_node_mat, N_diff_vec)
NTri_in_leafs_vec = find_tri_in_leafs(cen_tri_node_mat)
Triangle_normal_matrix = tri_normal_calculator_BVH_fkt(cen_tri_node_mat)

plot_bounding_volume(new_index_array, box_node_mat, cen_tri_node_mat, N_diff_vec, np.array([-7,7]), np.array([-7, 7]), np.array([0, 10]), 5, 30, 0)

BVH_data = (Triangle_normal_matrix, material_mat_node, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)


from range_histogram_data import remove_zeros_range, get_range_histogram
from setup_azimuthal_arc import azimuth_radar_location_fkt, save_distance_fkt
from loading_and_est_time import loading_bar, get_time_est
from remove_missed_rays import remove_missed_rays
import time


if get_4_pols == True:
    N_pols = 4
    pol_pars = ['HH', 'HV', 'VV', 'VH']
elif get_3_pols == True:
    N_pols = 3
    pol_pars = ['HH', 'HV', 'VV']
else:
    N_pols = 1
    pol_pars = pol_orientation


from pol_intersection_fkt import main_ray_BVH_Pol_BS_fkt, get_E_field_source

def main_loop_fkt(N_rays, beam_size, Radar_location, max_parameter_t, Detector_radius, pol_orientation):
    ray_origin_matrix, ray_direction_matrix, ray_E_field_matrix, trans_matrix = get_E_field_source(N_rays, pol_orientation, Radar_location, beam_size)
    
    distance_vec, weight_vec = main_ray_BVH_Pol_BS_fkt(N_bounces, N_rays, max_parameter_t, ray_origin_matrix, ray_direction_matrix, 
                                                    Radar_location, Detector_radius, 
                                                    Triangle_normal_matrix, material_mat_node,
                                                    new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,
                                                    ray_E_field_matrix, trans_matrix, pol_orientation)
    new_distance_vec, new_weight_vec = remove_missed_rays(distance_vec, weight_vec)
    return new_distance_vec, new_weight_vec




def SAR_raster(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
                   N_rays, beam_size, max_parameter_t, signal_res, pol_orientation):
    radar_loc_mat = azimuth_radar_location_fkt(slant_angle, radar_range,
                                   azimuth_steps, azimuth_start, azimuth_end)
    name_counter = 0
    tot_time = 0
    for azi in range(azimuth_steps):
        name_counter += 1
        time_1 = time.time()
        Rast_name_azi = "PolRT_azi_" + str(name_counter) + "_start_" + str(azimuth_start) + "_end_" + str(azimuth_end) + "_" + pol_name + '_' + pol_orientation
        Radar_location = radar_loc_mat[azi,:]
        
        distance_vec, weight_vec = main_loop_fkt(N_rays, beam_size, Radar_location, max_parameter_t, Detector_radius, pol_orientation)
        
        range_vec, range_his = get_range_histogram(distance_vec, weight_vec, max_parameter_t, signal_res)
        sparse_vec, sparse_his = remove_zeros_range(range_vec, range_his)
        Len_data = int(len(sparse_vec))
        save_data = np.zeros((Len_data,2))
        save_data[:,0], save_data[:,1] = sparse_vec, sparse_his
        save_distance_fkt(save_data, Rast_name_azi)
        
        loading_bar(name_counter, azimuth_steps)
        tot_time = get_time_est(time_1, tot_time, azimuth_steps, name_counter)


for n in range(N_pols):
    pol_orientation = pol_pars[n]
    print(pol_orientation)
    t1 = time.time()
    SAR_raster(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
                       N_rays, beam_size, max_parameter_t, signal_resolution, pol_orientation)
    print(time.time()- t1)


