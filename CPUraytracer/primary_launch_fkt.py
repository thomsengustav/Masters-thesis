# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 13:57:26 2024

@author: thoms
"""

#### primary launch of rays

'''
different methods for launcing rays used in ray-tracing and rasterization
'''

import numpy as np

from Hit_functions_fkt import hit_triangle_fkt, hit_sphere_fkt, hit_plane_fkt
from intersection_and_reflection_fkt import intercetion_point_fkt, plane_tri_ref_fkt, sphere_surface_norm_fkt
from ray_generation_function import launch_ray_components_fkt
from rotation_matrix_fkt import beam_direction_fkt, z_angle_fkt, rotation_matrix_fkt
from rotation_matrix_fkt import rotation_matrix_fkt

from numba import njit, prange


@njit
def secondary_spread_fkt(intersection_point, reflection_vec, ray_spread, N_multiply, surface_normal):
    rotation_matrix = rotation_matrix_fkt(reflection_vec)
    
    x, y = np.random.normal(0, ray_spread, N_multiply), np.random.normal(0, ray_spread, N_multiply)
    z = np.zeros(N_multiply)
    
    # preform rotation
    x_rot = x * rotation_matrix[0,0] + y * rotation_matrix[0,1] + z * rotation_matrix[0,2]
    y_rot = x * rotation_matrix[1,0] + y * rotation_matrix[1,1] + z * rotation_matrix[1,2]
    z_rot = x * rotation_matrix[2,0] + y * rotation_matrix[2,1] + z * rotation_matrix[2,2]
    
    Displacement_vector = intersection_point + reflection_vec
    x = x_rot + Displacement_vector[0]
    y = y_rot + Displacement_vector[1]
    z = z_rot + Displacement_vector[2]
    
    # calculate normalized vector components
    x_ray_component = x - intersection_point[0]
    y_ray_component = y - intersection_point[1] 
    z_ray_component = z - intersection_point[2] 
    
    # check for clipping!
    dot_x = x_ray_component * surface_normal[0]
    dot_y = y_ray_component * surface_normal[1]
    dot_z = z_ray_component * surface_normal[2]
    dot_res = dot_x + dot_y + dot_z
    
    zero_index = np.where(dot_res < 0)
    
    if len(zero_index[0]) > 0 and len(zero_index[0]) < N_multiply/2:
        zero_index = np.asarray(zero_index[0])
        x_ray_component_zero = np.delete(x_ray_component, zero_index)
        y_ray_component_zero = np.delete(y_ray_component, zero_index)
        z_ray_component_zero = np.delete(z_ray_component, zero_index)
        
        no_zeros = len(x_ray_component_zero)
        delta_rays = N_multiply - no_zeros
        noise_x, noise_y, noise_z = np.random.normal(0, 0.05, delta_rays), np.random.normal(0, 0.05, delta_rays), np.random.normal(0, 0.05, delta_rays)
        
        x_new = x_ray_component_zero[:delta_rays] + noise_x
        y_new = y_ray_component_zero[:delta_rays] + noise_y
        z_new = z_ray_component_zero[:delta_rays] + noise_z
        
        x_ray_component, y_ray_component, z_ray_component = np.zeros(N_multiply), np.zeros(N_multiply), np.zeros(N_multiply)
        x_ray_component[:no_zeros] = x_ray_component_zero
        y_ray_component[:no_zeros] = y_ray_component_zero
        z_ray_component[:no_zeros] = z_ray_component_zero
        x_ray_component[no_zeros:no_zeros+delta_rays] = x_new
        y_ray_component[no_zeros:no_zeros+delta_rays] = y_new
        z_ray_component[no_zeros:no_zeros+delta_rays] = z_new
    
    
    vector_length_inverse = 1 / np.sqrt(x_ray_component**2 + y_ray_component**2 + z_ray_component**2 )
    
    secondary_direction_mat = np.zeros((N_multiply,3))
    secondary_direction_mat[:,0] = np.multiply(x_ray_component, vector_length_inverse) 
    secondary_direction_mat[:,1] = np.multiply(y_ray_component, vector_length_inverse)
    secondary_direction_mat[:,2] = np.multiply(z_ray_component, vector_length_inverse)
    
    return secondary_direction_mat



@njit
def secondary_spread_method2_fkt(intersection_point, reflection_vec, ray_spread, N_multiply, surface_normal):
    x, y, z = np.random.normal(0, 1, N_multiply), np.random.normal(0, 1, N_multiply), np.random.normal(0, 1, N_multiply)

    vector_length_inverse = 1 / np.sqrt(x**2 + y**2 + z**2)
    alfa = 0.5
    
    diffuse_mat = np.zeros((N_multiply,3))
    diffuse_mat[:,0] = np.multiply(x, vector_length_inverse) + surface_normal[0]
    diffuse_mat[:,1] = np.multiply(y, vector_length_inverse) + surface_normal[1]
    diffuse_mat[:,2] = np.multiply(z, vector_length_inverse) + surface_normal[2]
    
    norm_diff = 1 / np.sqrt(diffuse_mat[:,0]**2 + diffuse_mat[:,1]**2 + diffuse_mat[:,2]**2)
    diffuse_mat[:,0] = np.multiply(diffuse_mat[:,0], norm_diff)
    diffuse_mat[:,1] = np.multiply(diffuse_mat[:,1], norm_diff)
    diffuse_mat[:,2] = np.multiply(diffuse_mat[:,2], norm_diff)
    
    secondary_direction_mat = np.zeros((N_multiply,3))
    secondary_direction_mat[:,0] = alfa * diffuse_mat[:,0] + (1 - alfa) * reflection_vec[0]
    secondary_direction_mat[:,1] = alfa * diffuse_mat[:,1] + (1 - alfa) * reflection_vec[1]
    secondary_direction_mat[:,2] = alfa * diffuse_mat[:,2] + (1 - alfa) * reflection_vec[2]
    
    return secondary_direction_mat


@njit
def calculate_numbers_of_objects_fkt(Triangle_normal_matrix, Plane_normal_matrix, Sphere_radius_vec):
    N_triangle = len(Triangle_normal_matrix[:,0])
    N_plane = len(Plane_normal_matrix[:,0])
    N_sphere = len(Sphere_radius_vec)
    Object_lenght_vector = np.array([N_triangle, N_plane, N_sphere])
    return Object_lenght_vector


@njit
def closest_object_primary_fkt(Triangle_matrix, Triangle_normal_matrix, 
                       Plane_point_matrix, Plane_normal_matrix,
                       Sphere_center_matrix, Sphere_radius, Object_lenght_vector, # load geometry
                       max_parameter_t, ray_origin, ray_direction):
    
    t = max_parameter_t
    surface_norm = np.zeros((3), dtype = float)
    for i in range(Object_lenght_vector[0]): # triangles
        hit_tri = hit_triangle_fkt(ray_origin, ray_direction, Triangle_matrix[0,:,i], Triangle_matrix[1,:,i], Triangle_matrix[2,:,i], max_parameter_t)
        if hit_tri == False or hit_tri > t:
            continue
        else:
            t = hit_tri
            surface_norm = Triangle_normal_matrix[i,:]
            
    for i in range(Object_lenght_vector[1]): # Planes
        hit_plane = hit_plane_fkt(ray_origin, ray_direction, Plane_point_matrix[i,:], Plane_normal_matrix[i,:], max_parameter_t)
        if hit_plane == False or hit_plane > t:
            continue
        else:
            t = hit_plane
            surface_norm = Plane_normal_matrix[i,:]
            
    for i in range(Object_lenght_vector[2]): # Spheres
        hit_sphere = hit_sphere_fkt(ray_origin, ray_direction, Sphere_center_matrix[i,:], Sphere_radius[i])
        if hit_sphere == False or hit_sphere > t:
            continue
        else:
            t = hit_sphere
            surface_norm = sphere_surface_norm_fkt(ray_origin, ray_direction, Sphere_center_matrix[i,:], Sphere_radius[i], t)
    
    if t >= max_parameter_t:
        return False, surface_norm # major jank! Should just return False
    
    return t, surface_norm
    

@njit
def check_normal_fkt(surface_normal, reflection_vec):
    dot =  np.dot(surface_normal, reflection_vec)
    if dot > 0:
        return surface_normal
    else:
        return -surface_normal

@njit(parallel=True)
def Primary_ray_iteration_fkt(N_primary, N_multiply, ray_origin_source, ray_direction_matrix, 
                             ray_spread, 
                             Triangle_matrix, Triangle_normal_matrix, 
                             Plane_point_matrix, Plane_normal_matrix,
                             Sphere_center_matrix, Sphere_radius, Object_lenght_vector, # load geometry
                             max_parameter_t):
    
    N_secondary = N_primary * N_multiply
    distance_vec = np.zeros(N_secondary)
    intersection_point_mat = np.zeros((N_secondary,3))
    reflection_vec_mat = np.zeros((N_secondary,3))
    for m in prange(N_primary):
        ray_origin = ray_origin_source
        ray_direction = ray_direction_matrix[m,:]
        parameter_object = closest_object_primary_fkt(Triangle_matrix, Triangle_normal_matrix, 
                               Plane_point_matrix, Plane_normal_matrix,
                               Sphere_center_matrix, Sphere_radius, Object_lenght_vector, # load geometry
                               max_parameter_t, ray_origin, ray_direction)
        if parameter_object[0] == False:
            continue
        t_parameter = parameter_object[0]
        suface_normal = parameter_object[1]
        intersection_point = intercetion_point_fkt(ray_origin, ray_direction, t_parameter)
        reflection_vec = plane_tri_ref_fkt(ray_direction, suface_normal)
        
        suface_normal = check_normal_fkt(suface_normal, reflection_vec)
        # secondary_direction_mat = secondary_spread_fkt(intersection_point, reflection_vec, ray_spread, N_multiply, suface_normal)
        secondary_direction_mat = secondary_spread_method2_fkt(intersection_point, reflection_vec, ray_spread, N_multiply, suface_normal)
        reflection_vec_mat[m*N_multiply:(m+1)*N_multiply, :] = secondary_direction_mat
        intersection_point_mat[m*N_multiply:(m+1)*N_multiply, :] = intersection_point
        distance_vec[m*N_multiply:(m+1)*N_multiply] = t_parameter
    
    return intersection_point_mat, reflection_vec_mat, distance_vec



@njit
def remove_missed_primary_rays_fkt(intersection_point_mat, reflection_vec_mat, distance_vec):
    new_distance_vec = distance_vec[distance_vec != 0]
    N_hits = len(new_distance_vec)
    
    new_intersection_point_mat = np.zeros((N_hits,3))
    new_reflection_vec_mat = np.zeros((N_hits,3))
    
    counter = 0
    for i in range(len(distance_vec)):
        if distance_vec[i] != 0:
            new_intersection_point_mat[counter,:] = intersection_point_mat[i,:]
            new_reflection_vec_mat[counter,:] = reflection_vec_mat[i,:]
            counter += 1
            
    return new_intersection_point_mat, new_reflection_vec_mat, new_distance_vec


    
