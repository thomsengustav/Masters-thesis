# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 13:26:04 2025

@author: thoms
"""

'''
Rasterization with no BVH for smaller scenes
'''

#### 'rasterization'-step

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import datetime
#### Initialize  parameters!

Rast_name = "tri_test_no_bvh_old" # name for saving data

###
radar_range = 100 # meters
slant_angle = 25 # degrees

azimuth_start = -5 # degrees
azimuth_end = 5 # degrees
azimuth_steps = 1

N_rays = 200000 # number of primary rays launched from source

t_max = 150 # longest allowed ray in meters

Source_spread = 0.1 # controls the spread of the radar beam
beam_spread = 0.1

max_tri_dist = 0.1

##### load geometry

from load_gmesh_square import load_gmesh_triangle_model
from load_model_center_rotate import load_big_model
from plot_vector_fkt import plot_mesh_and_normal
from better_mesh_fkt import better_mesh_fkt, calculate_normals
from rotate_mesh_fkt import rotate_mesh 
from fast_vec_operators import len_fkt, dot_fkt
from move_geometry_up import move_geometry_up

from intersection_and_reflection_fkt import normal_direction_check_fkt
# Triangle_matrix, Triangle_normal_matrix = load_big_model('t90a_trekanter') 
Triangle_matrix, Triangle_normal_matrix = load_gmesh_triangle_model('m_3_box.txt')

Triangle_matrix = move_geometry_up(Triangle_matrix)
Triangle_matrix = better_mesh_fkt(Triangle_matrix, max_tri_dist)
Triangle_normal_matrix = calculate_normals(Triangle_matrix)

# plot_mesh_and_normal(Triangle_matrix, Triangle_normal_matrix*0.1, np.array([-2, 2]), np.array([-2, 2]), np.array([-0.01, 5]), 30, 30, 0)


from fast_hit_tri import tri_hit_new, hit_triangle_fkt, hit_triangle_fkt_org

@njit
def closest_intersection_rast(N_tri, Triangle_matrix, ray_dir, ray_org, t_max):
    tri_index = -999
    t_min = 10000
    for i in range(N_tri):
        vertex0 = Triangle_matrix[0,:,i]
        vertex1 = Triangle_matrix[1,:,i]
        vertex2 = Triangle_matrix[2,:,i]
        # t = tri_hit_new(ray_dir, ray_org, vertex0, vertex1, vertex2)
        t = hit_triangle_fkt_org(ray_org, ray_dir, vertex0, vertex1, vertex2, t_max)
        
        if t < t_min:
            tri_index = i
            t_min = t
    
    return t_min, tri_index


@njit
def raster_int_fkt(N_rays, ray_origin_source, ray_direction_matrix, Triangle_matrix, Triangle_normal_matrix, t_max):
    distance_vec = np.zeros(N_rays)
    weight_vec = np.zeros(N_rays)
    ray_org = ray_origin_source
    N_tri =  int(len(Triangle_matrix[0,0,:]))
    for m in prange(N_rays):
        ray_dir = ray_direction_matrix[m,:]
        t_min, tri_index = closest_intersection_rast(N_tri, Triangle_matrix, ray_dir, ray_org, t_max)
        
        if t_min > t_max:
            continue
        
        suface_normal = Triangle_normal_matrix[tri_index,:]
        suface_normal = np.ascontiguousarray(suface_normal)
        intersection_point = ray_org + ray_dir * t_min
        
        ray_surface_dot, suface_normal = normal_direction_check_fkt(ray_dir, suface_normal)
        distance_vec[m] = 2*len_fkt(intersection_point - ray_origin_source)
        weight_vec[m] = dot_fkt(ray_dir, -suface_normal)
        
    return distance_vec, weight_vec


@njit
def remove_missed_rays(distance_vec, weight_vec):
    new_distance_vec = distance_vec[distance_vec != 0]
    N_hits = len(new_distance_vec)
    new_weight_vec = np.zeros((N_hits))
    
    counter = 0
    for i in range(len(distance_vec)):
        if distance_vec[i] != 0:
            new_weight_vec[counter] = weight_vec[i]
            counter += 1
            
    return new_distance_vec, new_weight_vec

from ray_generation_function import launch_ray_components_uniform_fkt
from rotation_matrix_fkt import beam_direction_fkt, rotation_matrix_fkt

@njit
def rast_sar_fkt(Radar_location, N_rays, Triangle_matrix, Triangle_normal_matrix, t_max):
    rotation_matrix = rotation_matrix_fkt(beam_direction_fkt(Radar_location))
    launch_ray_mat = np.zeros((N_rays,3))
    launch_ray_mat[:,0], launch_ray_mat[:,1], launch_ray_mat[:,2] = launch_ray_components_uniform_fkt(N_rays, beam_spread, rotation_matrix, Radar_location)
    
    distance_vec, weight_vec = raster_int_fkt(N_rays, Radar_location, launch_ray_mat, Triangle_matrix, Triangle_normal_matrix, t_max)
    
    new_distance_vec, new_weight_vec = remove_missed_rays(distance_vec, weight_vec)
    return new_distance_vec, new_weight_vec

from setup_azimuthal_arc import azimuth_radar_location_fkt, save_distance_fkt

def SAR_raster(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
                   N_rays, beam_spread, t_max, Triangle_matrix, Triangle_normal_matrix):
    radar_loc_mat = azimuth_radar_location_fkt(slant_angle, radar_range,
                                   azimuth_steps, azimuth_start, azimuth_end)
    name_counter = 0
    for azi in range(azimuth_steps):
        name_counter += 1
        Rast_name_azi = "rast_azi_" + str(name_counter) + "_start_" + str(azimuth_start) + "_end_" + str(azimuth_end) + "_" + Rast_name
        Radar_location = radar_loc_mat[azi,:]
        
        distance_vec, weight_vec = rast_sar_fkt(Radar_location, N_rays, Triangle_matrix, Triangle_normal_matrix, t_max)
        
        weight_vec = weight_vec**2
        # weight_vec = np.ones(int(len(weight_vec)))
        Len_data = int(len(distance_vec))
        save_data = np.zeros((Len_data,2))
        save_data[:,0], save_data[:,1] = distance_vec, weight_vec
        save_distance_fkt(save_data, Rast_name_azi)
        # print(name_counter)

import time

t1 = time.time()
SAR_raster(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
                   N_rays, beam_spread, t_max, Triangle_matrix, Triangle_normal_matrix)
print(time.time()- t1)




