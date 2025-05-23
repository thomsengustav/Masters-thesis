# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:21:07 2024

@author: thoms
"""

#### 'rasterization' step in RT

'''
The idea is to shot out rays at random and note the distance traveled back and forth.
These rays will not experience any physical scattering or difusion.
The intensity of the signal, as calculated from the resulting rays will be weighted arcording to
the dotprodiuct of the ray_dir and surface_norm.


note. This code has been used extesively for testing new functions and constructing different plots for the report.
The result is a very messy script...
'''

#### 'rasterization'-step

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import datetime
import time

#### Initialize  parameters!

Rast_name = "speed_rast_ref_4" # name for saving data

plot_footprint = True

###
radar_range = 100 # meters
slant_angle = 30 # degrees


azimuth_start = -4 # degrees
azimuth_end = 0 # degrees
azimuth_steps = 50

N_primary = 3000000 # number of primary rays launched from source

max_parameter_t = 150 # longest allowed ray in meters

Source_spread = 2 # controls the spread of the radar beam

max_in_leaf = 400 # tank
max_tri_dist = 1.5

signal_res = 0.001 # m
    
from loading_and_est_time import loading_bar, get_time_est

##### load geometry

from load_gmesh_square import load_gmesh_triangle_model
from load_model_center_rotate import load_big_model
from plot_vector_fkt import plot_mesh_and_normal
from better_mesh_fkt import better_mesh_fkt
from better_mesh_fkt import calculate_normals

# Triangle_matrix, Triangle_normal_matrix = load_big_model('t90a_trekanter') 
# Triangle_matrix, Triangle_normal_matrix = load_big_model('ford_model') 
# Triangle_matrix, Triangle_normal_matrix = load_big_model('råsted_vej3') 
# Triangle_matrix, Triangle_normal_matrix = load_gmesh_triangle_model('m_3_box.txt')

Triangle_matrix, Triangle_normal_matrix = load_big_model('speed_test_ref_4') 

Triangle_matrix = Triangle_matrix /  0.0254


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

def get_clutter_vec(Triangle_matrix, var2):
    N = int(len(Triangle_matrix[0,0,:]))
    clutter_vec = np.random.normal(1, var2, N)
    return clutter_vec


# function for placing geometry on imaginary plane at z=0
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
    
    return triangle_matrix

# Triangle_matrix = better_mesh_fkt(Triangle_matrix, max_tri_dist)

Triangle_matrix = move_geometry_up(Triangle_matrix)


clutter_vec = get_clutter_vec(Triangle_matrix, 0.5)
N = int(len(clutter_vec))
material_mat = np.zeros((2,N))
material_mat[0,:] = clutter_vec


@njit    
def get_avg_side_len(vertices):
    V1 = vertices[0,:]
    V2 = vertices[1,:]
    V3 = vertices[2,:]
    len1 = np.linalg.norm(V1-V2)
    len2 = np.linalg.norm(V1-V3)
    len3 = np.linalg.norm(V3-V2)
    avg_len = (len1+len2+len3)/3
    return avg_len

@njit
def get_displacemtn_xyz(normal, displacement_vec):
    dis_xyz = np.zeros((3,3))
    for i in range(3):
        dis_xyz[i,0] = normal[0] * displacement_vec[i]
        dis_xyz[i,1] = normal[1] * displacement_vec[i]
        dis_xyz[i,2] = normal[2] * displacement_vec[i]
    return dis_xyz

def add_plane_roughness(Triangle_matrix, roughness_par):
    # add random displacement to vertices perpendicular to triangle plane
    N = int(len(Triangle_matrix[0,0,:]))
    new_tri_mat = np.zeros((3,3,N))
    normals = calculate_normals(Triangle_matrix)
    for i in range(N):
        avg_len = get_avg_side_len(Triangle_matrix[:,:,i])
        displacement_vec = np.random.normal(0, roughness_par, 3)*avg_len
        dis_xyz = get_displacemtn_xyz(normals[i,:], displacement_vec)
        for n in range(3):
            for m in range(3):
                new_tri_mat[n,m,i] = dis_xyz[n,m] + Triangle_matrix[n,m,i]
    return new_tri_mat


    
# Triangle_matrix_r = add_plane_roughness(Triangle_matrix, 0.1)

Triangle_normal_matrix = calculate_normals(Triangle_matrix)


def get_rough_plane_tri_mat(Lenght, max_tri_dist, roughness_par):
    Triangle_matrix_new = np.zeros((3,3,2))
    # calculate vertex for 2 triangles
    V1, V2, V3 = np.array([-Lenght, -Lenght, 0]), np.array([-Lenght, Lenght, 0]), np.array([Lenght, -Lenght, 0])
    W1, W2, W3 = np.array([Lenght, Lenght, 0]), np.array([Lenght, -Lenght, 0]), np.array([-Lenght, Lenght, 0])
    Triangle_matrix_new[0,:,0] = V1
    Triangle_matrix_new[1,:,0] = V2
    Triangle_matrix_new[2,:,0] = V3
    Triangle_matrix_new[0,:,1] = W1
    Triangle_matrix_new[1,:,1] = W2
    Triangle_matrix_new[2,:,1] = W3
    # create fine mesh
    Triangle_matrix_fine = better_mesh_fkt(Triangle_matrix_new, max_tri_dist)
    # add roughness to mesh
    Triangle_matrix_rough = add_plane_roughness(Triangle_matrix_fine, roughness_par)
    # calculate surface normals
    Triangle_normal_matrix = calculate_normals(Triangle_matrix_rough)
    # get number of triangles in new mesh
    N = int(len(Triangle_matrix_rough[0,0,:]))
    return Triangle_matrix_rough, Triangle_normal_matrix, N

# Plane_tri, Plane_norms, N_plane = get_rough_plane_tri_mat(1, max_tri_dist, 0.1)

def add_rough_plane_to_tri(Triangle_matrix, Triangle_normal_matrix, Lenght, max_tri_dist, roughness_par):
    Plane_tri, Plane_norms, N_plane = get_rough_plane_tri_mat(Lenght, max_tri_dist, 0.1)
    N_old = int(len(Triangle_matrix[0,0,:]))
    Triangle_matrix_new = np.zeros((3,3,N_old+N_plane))
    Triangle_normal_matrix_new = np.zeros((N_old+N_plane,3))
    
    Triangle_matrix_new[:,:,:N_plane] = Plane_tri
    Triangle_matrix_new[:,:,N_plane:] = Triangle_matrix
    Triangle_normal_matrix_new[:N_plane,:] = Plane_norms
    Triangle_normal_matrix_new[N_plane:,:] = Triangle_normal_matrix
    
    return Triangle_matrix_new, Triangle_normal_matrix_new

#Triangle_matrix, Triangle_normal_matrix = add_rough_plane_to_tri(Triangle_matrix, Triangle_normal_matrix, 2, max_tri_dist, 0.1)
    
# plot_mesh_and_normal(Triangle_matrix, Triangle_normal_matrix*0.1, np.array([-5,5]), np.array([-5, 5]), np.array([-0.01, 10]), 30, 30, 0)

#### create BVH!

from BVH import BVH_builder, plot_bounding_volume, find_tri_in_leafs
from BVH import tri_normal_calculator_BVH_fkt, tree_node_fkt, optimize_index_array
from centroid_on_rasterization import BVH_builder_materials

box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec = BVH_builder(Triangle_matrix, max_in_leaf)
box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec, material_mat_node = BVH_builder_materials(Triangle_matrix, material_mat, max_in_leaf)

index_Array, N_diff_vec = tree_node_fkt(box_node_mat, cen_tri_node_mat, N_box_vec, parent_vec)
new_index_array = optimize_index_array(index_Array, box_node_mat, cen_tri_node_mat, N_diff_vec)

NTri_in_leafs_vec = find_tri_in_leafs(cen_tri_node_mat)

Triangle_normal_matrix = tri_normal_calculator_BVH_fkt(cen_tri_node_mat)

plot_bounding_volume(new_index_array, box_node_mat, cen_tri_node_mat, N_diff_vec, np.array([-5, 5]), np.array([-5, 5]), np.array([0, 10]), 30, 30, 0)

from BVH_ray_tracer_functions import raster_intersection_fkt, foot_print_intersection, foot_print_planewave_intersection, raster_intersection_PW_fkt 

from BVH_vol_overlap import getTotVolume, getTotOverlap

totVolVec = getTotVolume(box_node_mat)
print(totVolVec)

totOverlapVec = getTotOverlap(box_node_mat, totVolVec)
print(totOverlapVec)
####

from rotation_matrix_fkt import rotation_matrix_fkt, beam_direction_fkt
from ray_generation_function import launch_ray_components_fkt, launch_ray_components_uniform_fkt


def get_int_footprint(N_primary, beam_spread, Radar_location, max_parameter_t,
                      Triangle_matrix, Triangle_normal_matrix, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec):
    #generate rays from radar source
    rotation_matrix = rotation_matrix_fkt(beam_direction_fkt(Radar_location))
    launch_ray_mat = np.zeros((N_primary,3))
    launch_ray_mat[:,0], launch_ray_mat[:,1], launch_ray_mat[:,2] = launch_ray_components_uniform_fkt(N_primary, beam_spread, rotation_matrix, Radar_location)
    #return launch_ray_mat
    # launch primary rays
    intersection_points = foot_print_intersection(N_primary, Radar_location, launch_ray_mat, beam_spread, 
                                 Triangle_matrix, Triangle_normal_matrix, 
                                 new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                                 max_parameter_t)
    
    return intersection_points

def get_int_footprint_PW(N_primary, beam_spread, Radar_location, max_parameter_t,
                      Triangle_matrix, Triangle_normal_matrix, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec):
    #generate rays from radar source
    rotation_matrix = rotation_matrix_fkt(beam_direction_fkt(Radar_location))
    launch_ray_vec, Radar_location_mat = launch_ray_components_planewave_fkt(N_primary, beam_spread, rotation_matrix, Radar_location)
    print(Radar_location_mat[0:10,:])
    # launch primary rays
    
    intersection_points = foot_print_planewave_intersection(N_primary, Radar_location_mat, launch_ray_vec, beam_spread, 
                                 Triangle_matrix, Triangle_normal_matrix, 
                                 new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                                 max_parameter_t)
    return intersection_points

def get_plane_BVH_fkt(side_L):
    a = np.array([-side_L,side_L,0])
    b = np.array([-side_L,-side_L,0])
    c = np.array([side_L,side_L,0])
    d = np.array([side_L,-side_L,0])
    
    tri_mat = np.zeros((3,3,2))
    tri_mat[0,:,0] = a
    tri_mat[1,:,0] = b
    tri_mat[2,:,0] = d
    tri_mat[0,:,1] = a
    tri_mat[1,:,1] = c
    tri_mat[2,:,1] = d
    
    Triangle_matrix = tri_mat
    Triangle_matrix = better_mesh_fkt(Triangle_matrix, 2)
    Triangle_normal_matrix = calculate_normals(tri_mat)
    # plot_mesh_and_normal(Triangle_matrix, Triangle_normal_matrix*0.1, np.array([-10, 10]), np.array([-10, 10]), np.array([-3, 7]), 0, 30, 0)
    
    box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec = BVH_builder(Triangle_matrix, max_in_leaf)
    index_Array, N_diff_vec = tree_node_fkt(box_node_mat, cen_tri_node_mat, N_box_vec, parent_vec)
    new_index_array = optimize_index_array(index_Array, box_node_mat, cen_tri_node_mat, N_diff_vec)
    NTri_in_leafs_vec = find_tri_in_leafs(cen_tri_node_mat)
    Triangle_normal_matrix = tri_normal_calculator_BVH_fkt(cen_tri_node_mat)
    
    return Triangle_matrix, Triangle_normal_matrix, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec

def plot_footprint(N_primary, beam_spread, Radar_location, max_parameter_t, plane_L):
    Triangle_matrix, Triangle_normal_matrix, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec = get_plane_BVH_fkt(plane_L)
    points = get_int_footprint(N_primary, beam_spread, Radar_location, max_parameter_t,
                          Triangle_matrix, Triangle_normal_matrix, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)
    # points = get_int_footprint_PW(N_primary, beam_spread, Radar_location, max_parameter_t,
    #                            Triangle_matrix, Triangle_normal_matrix, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)
    plt.scatter(points[0,:], points[1,:], alpha=0.05)
    plt.xlim([-4,4])
    plt.ylim([-15,15])
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis('equal')
    plt.show()
    return points

@njit
def find_points(int_points, coord_index, minval, maxval):
    N = int(len(int_points[0,:]))
    point_vec = np.zeros((3,N))
    new_int_points = np.zeros((3,N))
    counter = 0
    for i in range(N):
        value = int_points[coord_index, i]
        if minval < value <= maxval:
            point_vec[:,counter] = int_points[:, i]
            counter+=1
        else:
            new_int_points[:,i-counter] = int_points[:, i]
    new_points = point_vec[:,:counter]
    new_int_points_n = new_int_points[:,:int(N-counter)]
    return new_points, new_int_points_n


@njit
def footprint_hits_fkt(intersection_points, dl, X, Y):
    dx = dy = dl
    Nx, Ny = int(X/dx), int(Y/dy)
    heatmap = np.zeros((Nx,Ny))
    for ix in range(Nx):
        minx = ix*dx - Nx*0.5*dx
        maxx = minx + dx
        x_vec, intersection_points = find_points(intersection_points, 0, minx, maxx)
        for iy in range(Ny):
            miny = iy*dy - Ny*0.5*dy
            maxy = miny + dy
            y_vec, x_vec = find_points(x_vec, 1, miny, maxy)
            count = int(len(y_vec[0,:]))
            heatmap[ix,iy] = count
    return heatmap

def get_heatmap_footprint(X, Y, dl, N_primary, beam_spread, Radar_location, max_parameter_t, plane_L):
    Triangle_matrix, Triangle_normal_matrix, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec = get_plane_BVH_fkt(plane_L)
    points = get_int_footprint(N_primary, beam_spread, Radar_location, max_parameter_t,
                          Triangle_matrix, Triangle_normal_matrix, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)
    # points = get_int_footprint_PW(N_primary, beam_spread, Radar_location, max_parameter_t,
    #                       Triangle_matrix, Triangle_normal_matrix, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)
    #return points
    heatmap = footprint_hits_fkt(points, dl, X, Y)
    #return heatmap, points
    Nx = int(len(heatmap[:,0]))
    Ny = int(len(heatmap[0,:]))
    xvec = np.linspace(-X/2, X/2, Nx+1)
    yvec = np.linspace(-Y/2, Y/2, Ny+1)
    plt.pcolormesh(xvec, yvec, np.transpose(heatmap))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.show()

from setup_azimuthal_arc import calculate_radar_loc
Radar_location = calculate_radar_loc(100, 25, 0)
if plot_footprint == True:
    get_heatmap_footprint(16, 35, 0.05, 500000, Source_spread, Radar_location, 150, 50)

from remove_missed_rays import remove_missed_rays
from centroid_on_rasterization import raster_intersection_centroid_fkt

@njit
def raster_main_fkt(N_primary, beam_spread, Radar_location, max_parameter_t):
    #generate rays from radar source
    rotation_matrix = rotation_matrix_fkt(beam_direction_fkt(Radar_location))
    launch_ray_mat = np.zeros((N_primary,3))
    launch_ray_mat[:,0], launch_ray_mat[:,1], launch_ray_mat[:,2] = launch_ray_components_uniform_fkt(N_primary, beam_spread, rotation_matrix, Radar_location)
    distance_vec, weight_vec = raster_intersection_centroid_fkt(N_primary, Radar_location, launch_ray_mat, beam_spread, 
                                 Triangle_matrix, Triangle_normal_matrix, 
                                 new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                                 max_parameter_t, material_mat_node)
    new_distance_vec, new_weight_vec = remove_missed_rays(distance_vec, weight_vec)
    return new_distance_vec, new_weight_vec

from setup_azimuthal_arc import azimuth_radar_location_fkt, save_distance_fkt

# bad results!
def add_dist_noise(distance_vec, noise_dist):
    N = int(len(distance_vec))
    random_dist = (np.random.rand(N) - 0.5) * noise_dist
    new_dist_vec = distance_vec + random_dist
    return new_dist_vec


from range_histogram_data import remove_zeros_range, get_range_histogram

def SAR_raster(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
                   N_primary, beam_spread, max_parameter_t):
    radar_loc_mat = azimuth_radar_location_fkt(slant_angle, radar_range,
                                   azimuth_steps, azimuth_start, azimuth_end)
    name_counter = 0
    tot_time = 0
    for azi in range(azimuth_steps):
        name_counter += 1
        time_1 = time.time()
        Rast_name_azi = "rast_azi_" + str(name_counter) + "_start_" + str(azimuth_start) + "_end_" + str(azimuth_end) + "_" + Rast_name
        #RT_name_azi = 'lol_test'
        Radar_location = radar_loc_mat[azi,:]
        
        distance_vec, weight_vec = raster_main_fkt(N_primary, beam_spread, Radar_location, max_parameter_t)
        
        weight_vec = np.ones(int(len(weight_vec)))
        
        range_vec, range_his = get_range_histogram(distance_vec, weight_vec, max_parameter_t, signal_res)
        sparse_vec, sparse_his = remove_zeros_range(range_vec, range_his)
        Len_data = int(len(sparse_vec))
        save_data = np.zeros((Len_data,2))
        save_data[:,0], save_data[:,1] = sparse_vec, sparse_his
        
        save_distance_fkt(save_data, Rast_name_azi)
        loading_bar(name_counter, azimuth_steps)
        tot_time = get_time_est(time_1, tot_time, azimuth_steps, name_counter)

t1 = time.time()
SAR_raster(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
                   N_primary, Source_spread, max_parameter_t)
print(time.time()- t1)


# #####
# '''
# run D_max and k sweep!
# '''


# @njit
# def raster_main_fkt2(N_primary, beam_spread, Radar_location, max_parameter_t, 
#                     Triangle_matrix, Triangle_normal_matrix, 
#                     new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec):
#     #generate rays from radar source
#     rotation_matrix = rotation_matrix_fkt(beam_direction_fkt(Radar_location))
#     launch_ray_mat = np.zeros((N_primary,3))
#     launch_ray_mat[:,0], launch_ray_mat[:,1], launch_ray_mat[:,2] = launch_ray_components_uniform_fkt(N_primary, beam_spread, rotation_matrix, Radar_location)
#     distance_vec, weight_vec = raster_intersection_fkt(N_primary, Radar_location, launch_ray_mat, beam_spread, 
#                                   Triangle_matrix, Triangle_normal_matrix, 
#                                   new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
#                                   max_parameter_t)
    
#     new_distance_vec, new_weight_vec = remove_missed_rays(distance_vec, weight_vec)
#     return new_distance_vec, new_weight_vec

# def SAR_raster2(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
#                    N_primary, beam_spread, max_parameter_t, Triangle_matrix, Triangle_normal_matrix, 
#                    new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec):
#     radar_loc_mat = azimuth_radar_location_fkt(slant_angle, radar_range,
#                                    azimuth_steps, azimuth_start, azimuth_end)
#     name_counter = 0
#     tot_time = 0
#     for azi in range(azimuth_steps):
#         name_counter += 1
#         time_1 = time.time()
#         Rast_name_azi = "rast_azi_" + str(name_counter) + "_start_" + str(azimuth_start) + "_end_" + str(azimuth_end) + "_" + Rast_name
#         #RT_name_azi = 'lol_test'
#         Radar_location = radar_loc_mat[azi,:]
        
#         distance_vec, weight_vec = raster_main_fkt2(N_primary, beam_spread, Radar_location, max_parameter_t, 
#                             Triangle_matrix, Triangle_normal_matrix, 
#                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)
        
#         weight_vec = np.ones(int(len(weight_vec)))
        
#         range_vec, range_his = get_range_histogram(distance_vec, weight_vec, max_parameter_t, signal_res)
#         sparse_vec, sparse_his = remove_zeros_range(range_vec, range_his)
#         Len_data = int(len(sparse_vec))
#         save_data = np.zeros((Len_data,2))
#         save_data[:,0], save_data[:,1] = sparse_vec, sparse_his
        
#         save_distance_fkt(save_data, Rast_name_azi)
#         loading_bar(name_counter, azimuth_steps)
#         tot_time = get_time_est(time_1, tot_time, azimuth_steps, name_counter)


# SAR_raster2(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
#                     20, Source_spread, max_parameter_t, Triangle_matrix, Triangle_normal_matrix, 
#                     new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)


# org_tri_mat, org_tri_norm_mat = Triangle_matrix, Triangle_normal_matrix = load_big_model('t90a_trekanter') 

# dist_max = np.linspace(1.4, 0.2, 13)#14)
# k_lin = np.linspace(13, 14, 2)#12)

# #TODO
# #13-14
# # 0.1

# time_mat = np.zeros((int(len(dist_max)), int(len(k_lin))))

# for i in range(int(len(dist_max))):
#     max_tri_dist = dist_max[i]
#     print(max_tri_dist)
#     Triangle_matrix = better_mesh_fkt(org_tri_mat, max_tri_dist)
#     Triangle_matrix = move_geometry_up(Triangle_matrix)
#     Triangle_normal_matrix = calculate_normals(Triangle_matrix)

#     N_tris = len(Triangle_normal_matrix[:,0])
#     for n in range(int(len(k_lin))):
#         k = k_lin[n]
#         max_in_leaf = int(np.ceil(N_tris / (2**k))) + 1
#         print(max_in_leaf)
        
#         box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec = BVH_builder(Triangle_matrix, max_in_leaf)
#         index_Array, N_diff_vec = tree_node_fkt(box_node_mat, cen_tri_node_mat, N_box_vec, parent_vec)
#         new_index_array = optimize_index_array(index_Array, box_node_mat, cen_tri_node_mat, N_diff_vec)
#         NTri_in_leafs_vec = find_tri_in_leafs(cen_tri_node_mat)
#         Triangle_normal_matrix = tri_normal_calculator_BVH_fkt(cen_tri_node_mat)
#         # # plot_bounding_volume(new_index_array, box_node_mat, cen_tri_node_mat, N_diff_vec, np.array([-5, 5]), np.array([-5, 5]), np.array([0, 10]), 30, 30, 0)
        
        
#         t1 = time.time()
#         SAR_raster2(slant_angle, radar_range, azimuth_steps, azimuth_start, azimuth_end, 
#                             N_primary, Source_spread, max_parameter_t, Triangle_matrix, Triangle_normal_matrix, 
#                             new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec# load geometry
#                             )
#         timenum = time.time() - t1  
#         time_mat[i,n] = timenum
