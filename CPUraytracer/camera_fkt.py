# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 09:15:01 2024

@author: thoms
"""

'''
rasterization of meshes to create images of the models
'''


import numpy as np
from numba import njit, float64

### load geometry

max_in_leaf = 370 # tank
max_tri_dist = 0.4

##### load geometry

from load_gmesh_square import load_gmesh_triangle_model
from load_model_center_rotate import load_big_model
from plot_vector_fkt import plot_mesh_and_normal
from better_mesh_fkt import better_mesh_fkt

# Triangle_matrix, Triangle_normal_matrix = load_big_model('t90a_trekanter')
# Triangle_matrix, Triangle_normal_matrix = load_gmesh_triangle_model('m_3_box.txt')
# Triangle_matrix, Triangle_normal_matrix = load_gmesh_triangle_model('sphere4.txt')
Triangle_matrix, Triangle_normal_matrix = load_big_model('ford_model') 

Triangle_matrix = Triangle_matrix*35


def rotate_ford(Triangle_matrix):
    N = int(len(Triangle_matrix[0,0,:]))
    new_Triangle_matrix = np.zeros((3,3,N))
    new_Triangle_matrix[:,0,:] = Triangle_matrix[:,0,:]
    new_Triangle_matrix[:,1,:] = Triangle_matrix[:,1,:]
    new_Triangle_matrix[:,2,:] = -Triangle_matrix[:,2,:]
    return new_Triangle_matrix


Triangle_matrix = rotate_ford(Triangle_matrix)

from rotate_mesh_fkt import rotate_mesh

Triangle_matrix = rotate_mesh(Triangle_matrix, 0)

#Triangle_matrix = better_mesh_fkt(Triangle_matrix, max_tri_dist)

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
    
    return Triangle_matrix

Triangle_matrix = move_geometry_up(Triangle_matrix)

#plot_mesh_and_normal(Triangle_matrix[:,:,:10000], Triangle_normal_matrix[:10000,:]*0.1, np.array([-5, 5]), np.array([-5, 5]), np.array([0, 10]), 30, 30, 0)
#### create BVH!

from BVH import BVH_builder, plot_bounding_volume, find_tree_ray_intersection_fkt, find_tri_in_leafs
from BVH import tri_normal_calculator_BVH_fkt, tree_node_fkt, optimize_index_array, plot_bounding_volume

box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec = BVH_builder(Triangle_matrix, max_in_leaf)
index_Array, N_diff_vec = tree_node_fkt(box_node_mat, cen_tri_node_mat, N_box_vec, parent_vec)
new_index_array = optimize_index_array(index_Array, box_node_mat, cen_tri_node_mat, N_diff_vec)
NTri_in_leafs_vec = find_tri_in_leafs(cen_tri_node_mat)
Triangle_normal_matrix = tri_normal_calculator_BVH_fkt(cen_tri_node_mat)
plot_bounding_volume(new_index_array, box_node_mat, cen_tri_node_mat, N_diff_vec, np.array([-5, 5]), np.array([-5, 5]), np.array([0, 10]), 30, 30, 0)
    

from rotation_matrix_fkt import beam_direction_fkt, z_angle_fkt, rotation_matrix_fkt

@njit
def get_xy_cam_pos(Nx, Ny):
    x = np.zeros(Nx*Ny)
    y = np.zeros(Nx*Ny)
    pix_size = 0.005
    Lx = pix_size * Nx
    Ly = pix_size * Ny
    counter = 0
    for i in range(Nx):
        x_pos = i*pix_size - Lx / 2 + pix_size / 2
        for n in range(Ny):
            y_pos = n*pix_size - Ly / 2 + pix_size / 2
            
            x[counter] = x_pos
            y[counter] = y_pos
            counter += 1
    return x, y

x, y = get_xy_cam_pos(400, 400)


@njit
def launch_ray_camera_fkt(Nx, Ny, fov_len, look_dir, cam_pos, rotation_matrix):
    x, y = get_xy_cam_pos(Nx, Ny)
    
    z = np.zeros(Nx*Ny)
    
    # preform rotation
    x_rot = x * rotation_matrix[0,0] + y * rotation_matrix[0,1] + z * rotation_matrix[0,2]
    y_rot = x * rotation_matrix[1,0] + y * rotation_matrix[1,1] + z * rotation_matrix[1,2]
    z_rot = x * rotation_matrix[2,0] + y * rotation_matrix[2,1] + z * rotation_matrix[2,2]
    
    Displacement_vector = cam_pos + look_dir * fov_len
    x = x_rot + Displacement_vector[0]
    y = y_rot + Displacement_vector[1]
    z = z_rot + Displacement_vector[2]
    
    # calculate normalized vector components
    x_ray_component = x - cam_pos[0]
    y_ray_component = y - cam_pos[1] 
    z_ray_component = z - cam_pos[2] 
    
    vector_length_inverse = 1 / np.sqrt(x_ray_component**2 + y_ray_component**2 + z_ray_component**2 )
    
    x_ray_component_norm = np.multiply(x_ray_component, vector_length_inverse)  
    y_ray_component_norm = np.multiply(y_ray_component, vector_length_inverse)
    z_ray_component_norm = np.multiply(z_ray_component, vector_length_inverse)
    
    return x_ray_component_norm, y_ray_component_norm, z_ray_component_norm


from BVH_ray_tracer_functions import raster_intersection_fkt

# assumes detector location = radar location
def raster_cam_fkt(Nx, Ny, fov_len, look_dir, cam_pos, max_parameter_t, rotation_matrix):
    #generate rays from radar source
    ray_org = cam_pos - look_dir * fov_len
    launch_ray_mat = np.zeros((Nx*Ny,3))
    launch_ray_mat[:,0], launch_ray_mat[:,1], launch_ray_mat[:,2] = launch_ray_camera_fkt(Nx, Ny, fov_len, look_dir, cam_pos, rotation_matrix)
    # launch primary rays
    N_primary = Nx*Ny
    distance_vec, weight_vec = raster_intersection_fkt(N_primary, ray_org, launch_ray_mat, 1,
                                 Triangle_matrix, Triangle_normal_matrix, 
                                 new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec,# load geometry
                                 max_parameter_t)
    weight_vec = weight_vec
    #weight_vec = np.ones(int(len(weight_vec)))
    return distance_vec, weight_vec

@njit
def set_weights_to_zero(distance_vec, weight_vec):
    N_hits = len(distance_vec)
    new_weight_vec = weight_vec
    
    for i in range(len(distance_vec)):
        if distance_vec[i] == 0:
            new_weight_vec[i] = 0
            
    return new_weight_vec


def make_image_mat(Nx, Ny, weight_vec):
    image_mat = np.zeros((Nx,Ny))
    counter = 0
    for i in range(Nx):
        for n in range(Ny):
            image_mat[i,n] = weight_vec[counter]
            counter += 1
    return image_mat

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

def camera_fkt(Nx, Ny, fov_len, look_dir, cam_pos, max_parameter_t):
    rotation_matrix = rotation_matrix_fkt(look_dir)
    distance_vec, weight_vec = raster_cam_fkt(Nx, Ny, fov_len, look_dir, cam_pos, max_parameter_t, rotation_matrix)
    new_weights_vec = set_weights_to_zero(distance_vec, weight_vec)
    image_mat = make_image_mat(Nx, Ny, weight_vec)
    
    plt.imshow(image_mat, cmap=mpl.colormaps['gray'])
    # plt.axis('equal')
    # plt.colorbar()
    # plt.show()
    
    # plt.pcolormesh(xvec, yvec, np.transpose(heatmap))
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')
    plt.colorbar()
    plt.show()

# camera setup
direct = np.array([0, 1, 0])
direct = direct / np.linalg.norm(direct)
np.array([-2.5,-1.6,3])

camera_fkt(600, 800, 2, direct, np.array([0,-4,1]), 150)
            