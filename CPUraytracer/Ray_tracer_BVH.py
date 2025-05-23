# -*- coding: utf-8 -*-
"""
Created on Tue oct 7

@author: thoms
"""

#### main ray tracer!

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import datetime
#### Initialize ray tracer parameters!

RT_name = "sphere4_rt_vs_rast_test" # name for saving data

###
radar_range = 100 # meters
slant_angle = 30 # degrees

azimuth_start = -5 # degrees
azimuth_end = 5 # degrees
azimuth_steps = 1

N_primary = 100000 # number of primary rays launched from source
N_secondary = 10 # number of secondary rays generated from primary ray-surface intersection

N_bounces = 3 # number of allowed bounces
max_parameter_t = 150 # longest allowed ray in meters

Source_spread = 0.2 # controls the spread of the radar beam
secondary_spread = 0.5 # controls the spread of secondary rays from primary ray-surface intersection, basically 'diffussion'.

Detector_radius = 15 # meters

max_in_leaf = 370 # tank
max_tri_dist = 0.4
# max_in_leaf = 500 # sphere4 best!

def save_parameters(radar_range, slant_angle, azimuth_start, azimuth_end, azimuth_steps,
                    N_primary, N_secondary, N_bounces, max_parameter_t, Source_spread, secondary_spread,
                    Detector_radius, max_in_leaf, max_tri_dist, RT_name):
    par = []
    par.append('Radar range = ' + str(radar_range))
    par.append('Slant angle = ' + str(slant_angle))
    par.append('Azimuth start = ' + str(azimuth_start))
    par.append('Azimuth end = ' + str(azimuth_end))
    par.append('Azimuth steps = ' + str(azimuth_steps))
    par.append('Number of primary rays = ' + str(N_primary))
    par.append('Number of secondary rays = ' + str(N_secondary))
    par.append('Number of allowed bounces = ' + str(N_bounces))
    par.append('Maximum allowed parameter t = ' + str(max_parameter_t))
    par.append('Source spread = ' + str(Source_spread))
    par.append('Secondary spread = ' + str(secondary_spread))
    par.append('Detector radius = ' + str(Detector_radius))
    par.append('Maximum allowed triangles in leafs = ' + str(max_in_leaf))
    par.append('Maximum allowed triangle parameter = ' + str(max_tri_dist))
    
    string = 'Ray tracer parameters:'
    for i in range(len(par)):
        string = string + '\n' + par[i]
    
    current_time = datetime.datetime.now()
    date = current_time.strftime(" %m%d%Y %H,%M")
    name = 'RT_parameters' + date + ' '+ RT_name + '.txt'
    f = open(name, "w")
    f.write(string)
    f.close()

save_parameters(radar_range, slant_angle, azimuth_start, azimuth_end, azimuth_steps,
                    N_primary, N_secondary, N_bounces, max_parameter_t, Source_spread, secondary_spread,
                    Detector_radius, max_in_leaf, max_tri_dist, RT_name)

##### load geometry

from point_scater_triangle import Point_scatter_fkt
from load_gmesh_square import load_gmesh_triangle_model
from load_model_center_rotate import load_big_model
from plot_vector_fkt import plot_mesh_and_normal
from better_mesh_fkt import better_mesh_fkt

# initializing geometry
Triangle_matrix = np.zeros((3,3,1))
Triangle_normal_matrix = np.zeros((1,3))
Plane_point_matrix = np.zeros((1,3))
Plane_normal_matrix = np.zeros((1,3))
Sphere_center_matrix = np.zeros((1,3))
Sphere_radius_vec = np.zeros(1)

# load mesh
Triangle_matrix, Triangle_normal_matrix = load_gmesh_triangle_model('sphere4.txt')
# Triangle_matrix, Triangle_normal_matrix = Point_scatter_fkt(2, np.array([0.,0.,0.]))
# Triangle_matrix, Triangle_normal_matrix = load_big_model('t90a_trekanter')

Triangle_matrix, Triangle_normal_matrix = load_big_model('speed_test_ref_3') 

Triangle_matrix = Triangle_matrix /  0.0254

# Triangle_matrix = np.load('t90a_tri_mat.npy')
# Triangle_normal_matrix = np.load('t90a_norm_mat.npy')

from better_mesh_fkt import better_mesh_fkt
        
# Triangle_matrix = better_mesh_fkt(Triangle_matrix, max_tri_dist)

def rotate_ford(Triangle_matrix):
    N = int(len(Triangle_matrix[0,0,:]))
    new_Triangle_matrix = np.zeros((3,3,N))
    new_Triangle_matrix[:,0,:] = Triangle_matrix[:,0,:]
    new_Triangle_matrix[:,1,:] = Triangle_matrix[:,2,:]
    new_Triangle_matrix[:,2,:] = Triangle_matrix[:,1,:]
    return new_Triangle_matrix
Triangle_matrix = rotate_ford(Triangle_matrix)

from rotate_mesh_fkt import rotate_mesh

Triangle_matrix = rotate_mesh(Triangle_matrix, 35)

# # function for placing geometry on imaginary plane at z=0
def move_geometry_up(triangle_matrix):
    N_triangle = int(len(triangle_matrix[0,0,:]))
    
    # center model xy at 0,0
    x_vec = np.zeros(N_triangle*3)
    y_vec = np.zeros(N_triangle*3)
    z_vec = np.zeros(N_triangle*3)
    for i in range(N_triangle):
        x_point = triangle_matrix[:,0,i]
        x_vec[i:i+3] = x_point
        y_point = triangle_matrix[:,1,i]
        y_vec[i:i+3] = y_point
        z_point = triangle_matrix[:,2,i]
        z_vec[i:i+3] = z_point
    x_avg = np.sum(x_vec)/N_triangle
    y_avg = np.sum(y_vec)/N_triangle
    min_z = min(z_vec)
    
    triangle_matrix[:,0,:] = triangle_matrix[:,0,:] - x_avg
    triangle_matrix[:,1,:] = triangle_matrix[:,1,:] - y_avg
    triangle_matrix[:,2,:] = triangle_matrix[:,2,:] - min_z
    
    return Triangle_matrix

Triangle_matrix = move_geometry_up(Triangle_matrix)

@njit
def calculate_normals(tri_mat):
    N_triangle = int(len(tri_mat[0,0,:]))
    triangle_normal_matrix = np.zeros((N_triangle,3))
    for i in range(N_triangle):
        V1 = tri_mat[0,:,i]
        V2 = tri_mat[1,:,i]
        V3 = tri_mat[2,:,i]

        norm = np.cross(V1-V3,V2-V3)
        norm = norm / np.linalg.norm(norm)
        triangle_normal_matrix[i,:] = norm
    return triangle_normal_matrix

Triangle_normal_matrix = calculate_normals(Triangle_matrix)

plot_mesh_and_normal(Triangle_matrix[:,:,:], Triangle_normal_matrix[:,:]*0.1, np.array([-5, 5]), np.array([-5, 5]), np.array([0, 10]), 0, 30, 0)
#### create BVH!

from BVH import BVH_builder, plot_bounding_volume, find_tri_in_leafs
from BVH import tri_normal_calculator_BVH_fkt, tree_node_fkt, optimize_index_array

box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec = BVH_builder(Triangle_matrix, max_in_leaf)

index_Array, N_diff_vec = tree_node_fkt(box_node_mat, cen_tri_node_mat, N_box_vec, parent_vec)
new_index_array = optimize_index_array(index_Array, box_node_mat, cen_tri_node_mat, N_diff_vec)

NTri_in_leafs_vec = find_tri_in_leafs(cen_tri_node_mat)

Triangle_normal_matrix = tri_normal_calculator_BVH_fkt(cen_tri_node_mat)

plot_bounding_volume(new_index_array, box_node_mat, cen_tri_node_mat, N_diff_vec, np.array([-5, 5]), np.array([-5, 5]), np.array([0, 10]), 30, 30, 0)

from BVH_ray_tracer_functions import Primary_ray_iteration_BVH_fkt, main_ray_iteration_BVH_fkt, main_ray_iteration_BVH_test_fkt 
from primary_launch_fkt import remove_missed_primary_rays_fkt

####

from rotation_matrix_fkt import rotation_matrix_fkt, beam_direction_fkt
from ray_generation_function import launch_ray_components_fkt
from closest_intersection_fkt import main_ray_iteration_fkt, remove_zeros_fkt

# assumes detector location = radar location
def Ray_tracer_fkt(N_primary, N_secondary, N_bounces, beam_spread, secondary_spread, Radar_location, max_parameter_t,
                   Detector_radius):
    #generate rays from radar source
    rotation_matrix = rotation_matrix_fkt(beam_direction_fkt(Radar_location))
    launch_ray_mat = np.zeros((N_primary,3))
    launch_ray_mat[:,0], launch_ray_mat[:,1], launch_ray_mat[:,2] = launch_ray_components_fkt(N_primary, beam_spread, rotation_matrix, Radar_location)
    #return launch_ray_mat
    # launch primary rays
    intersection_point_mat, reflection_vec_mat, distance_vec = Primary_ray_iteration_BVH_fkt(N_primary, N_secondary, Radar_location, launch_ray_mat, 
                                                                                          secondary_spread, 
                                                                                          Triangle_matrix, Triangle_normal_matrix, 
                                                                                          new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec, # load geometry
                                                                                          max_parameter_t)
    print(len(distance_vec))
    intersection_point_mat, reflection_vec_mat, distance_vec = remove_missed_primary_rays_fkt(intersection_point_mat, reflection_vec_mat, distance_vec)
    # main ray tracer loop
    N_rays = len(distance_vec)
    print(N_rays)
    distance_vec_final = main_ray_iteration_BVH_fkt(N_bounces, N_rays, max_parameter_t, intersection_point_mat, reflection_vec_mat, 
                                  Radar_location, Detector_radius, distance_vec, Triangle_matrix, Triangle_normal_matrix, 
                                  new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)
    distance_vec_final = remove_zeros_fkt(distance_vec_final)
    #print(distance_vec_final)
    return distance_vec_final


def plot_rays_in_scene(plot_rays_mat, Radar_location):
    N_rays = int(len(plot_rays_mat[0,:,0]))
    N_bounces = int(len(plot_rays_mat[0,0,:]))
    
    for i in range(N_rays):
        x = np.array([Radar_location[0]])
        y = np.array([Radar_location[1]])
        z = np.array([Radar_location[2]])
        ax = plt.figure().add_subplot(projection='3d')
        for n in range(N_bounces):
            xn = plot_rays_mat[0,i,n]
            yn = plot_rays_mat[1,i,n]
            zn = plot_rays_mat[2,i,n]
            if xn == 0 and yn == 0 and zn == 0:
                continue
            else:
                x = np.append(x, xn)
               # print(x)
                y = np.append(y, yn)
                z = np.append(z, zn)
        ax.plot(x,y,z)
        ax.set_xlim(-25, 25)
        ax.set_zlim(-2, 40)
        ax.set_ylim(-1, 100)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.view_init(elev=30, azim=180, roll=0)
        plt.show()

# plot_rays_in_scene(plot_rays_mat, Radar_location)

# limit memeory use by splitting up number of primary rays pr. iteration.

def primary_ray_splitter(N_primary):
    counter = 0
    for i in range(200):
        N_new = N_primary / (i+1)
        counter += 1
        if N_new <= 200000:
            N_new = int(N_new)
            return counter, N_new


def Final_Ray_tracer_fkt(N_primary, N_secondary, N_bounces, beam_spread, secondary_spread, Radar_location, max_parameter_t,
                   Detector_radius):
    global_distance_vec = np.zeros((1))
    ray_it, N_P_new =  primary_ray_splitter(N_primary)
    for i in range(ray_it):
        distance_it = Ray_tracer_fkt(N_P_new, N_secondary, N_bounces, Source_spread, secondary_spread, Radar_location, max_parameter_t, Detector_radius)
        global_distance_vec = np.append(global_distance_vec, distance_it)
    
    global_distance_vec = global_distance_vec[1:]
    return global_distance_vec 

###

from setup_azimuthal_arc import azimuth_radar_location_fkt, save_distance_fkt

def SAR_ray_tracer(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
                   N_primary, N_secondary, N_bounces, beam_spread, secondary_spread, max_parameter_t, Detector_radius):
    radar_loc_mat = azimuth_radar_location_fkt(slant_angle, radar_range,
                                   azimuth_steps, azimuth_start, azimuth_end)
    name_counter = 0
    for azi in range(azimuth_steps):
        name_counter += 1
        RT_name_azi = "azi_" + str(name_counter) + "_start_" + str(azimuth_start) + "_end_" + str(azimuth_end) + "_" + RT_name
        #RT_name_azi = 'lol_test'
        Radar_location = radar_loc_mat[azi,:]
        
        distance_vec = Final_Ray_tracer_fkt(N_primary, N_secondary, N_bounces, beam_spread, secondary_spread, Radar_location, max_parameter_t,
                           Detector_radius)
        
        save_distance_fkt(distance_vec, RT_name_azi)
        print(name_counter)

###### run the ray tracer

import time

t1 = time.time()
SAR_ray_tracer(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
                       N_primary, N_secondary, N_bounces, Source_spread, secondary_spread, max_parameter_t, Detector_radius)
print(time.time()- t1)



