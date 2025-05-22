# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:57:20 2024

@author: thoms
"""

'''
very old code!
'''


#### Find closest intersection of all scene objects

import numpy as np
from Hit_functions_fkt import hit_triangle_fkt, hit_sphere_fkt, hit_plane_fkt, hit_triangle_par_fkt
from numba import njit, prange

@njit
def closest_object_tri_fkt(Triangle_matrix, Triangle_normal_matrix, 
                       Plane_point_matrix, Plane_normal_matrix,
                       Sphere_center_matrix, Sphere_radius, Object_lenght_vector,
                       det_center, det_radius, # load geometry
                       max_parameter_t, ray_origin, ray_direction):
    
    t = max_parameter_t
    surface_norm = np.zeros((3), dtype = float)
    
    hit_tri = hit_triangle_par_fkt(ray_origin, ray_direction, Triangle_matrix[0,:,:], Triangle_matrix[1,:,:], Triangle_matrix[2,:,:], max_parameter_t)
    if hit_tri[0] != False:
        t = hit_tri[0]
        surface_norm = Triangle_normal_matrix[hit_tri[1],:]
    
            
    # for i in range(Object_lenght_vector[1]): # Planes
    #     hit_plane = hit_plane_fkt(ray_origin, ray_direction, Plane_point_matrix[i,:], Plane_normal_matrix[i,:], max_parameter_t)
    #     if hit_plane == False or hit_plane > t:
    #         continue
    #     else:
    #         t = hit_plane
    #         surface_norm = Plane_normal_matrix[i,:]
            
    # for i in range(Object_lenght_vector[2]): # Spheres
    #     hit_sphere = hit_sphere_fkt(ray_origin, ray_direction, Sphere_center_matrix[i,:], Sphere_radius[i])
    #     if hit_sphere == False or hit_sphere > t:
    #         continue
    #     else:
    #         t = hit_sphere
    #         surface_norm = sphere_surface_norm_fkt(ray_origin, ray_direction, Sphere_center_matrix[i,:], Sphere_radius[i], t)
    
    hit_det = hit_sphere_fkt(ray_origin, ray_direction, det_center, det_radius)
    if hit_det < t and hit_det > 0:
        t = hit_det
            
    if t == max_parameter_t:
        return False, surface_norm
    else:
        return t, surface_norm


@njit
def closest_object_fkt(Triangle_matrix, Triangle_normal_matrix, 
                       Plane_point_matrix, Plane_normal_matrix,
                       Sphere_center_matrix, Sphere_radius, Object_lenght_vector,
                       det_center, det_radius, # load geometry
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
    
    hit_det = hit_sphere_fkt(ray_origin, ray_direction, det_center, det_radius)
    if hit_det < t and hit_det > 0:
        t = hit_det
            
    if t == max_parameter_t:
        return False, surface_norm
    else:
        return t, surface_norm
    


from intersection_and_reflection_fkt import intercetion_point_fkt, plane_tri_ref_fkt, sphere_surface_norm_fkt
### main loop ish
import matplotlib.pyplot as plt

def plot_beam_profile(launch_ray_mat, Radar_location):
    N_rays = int(len(launch_ray_mat[:,0]))
    ray_origin = Radar_location
    X_profile = np.zeros(N_rays)
    Y_profile = np.zeros(N_rays)
    for i in range(N_rays):
        ray_direction = launch_ray_mat[i,:]
        t = hit_plane_fkt(ray_origin, ray_direction, np.array([0.,0.,0]), np.array([0.,0.,1.]), 2500)
        int_point = intercetion_point_fkt(ray_origin, ray_direction, t)
        X_profile[i] = int_point[0]
        Y_profile[i] = int_point[1]
    plt.scatter(X_profile,Y_profile, alpha = 0.01)
    #plt.axis('equal')
    plt.xlim(-10, 10)
    plt.ylim(20, -30)
    plt.show()
    
        
@njit
def detector_check_fkt(intersection_point, det_center, det_radius):
    new_vec = intersection_point - det_center
    lenght_new_vec = np.linalg.norm(new_vec)
    if lenght_new_vec < det_radius * 1.001:
        return True
    else:
        return False
    

@njit
def detector_lenght_cal_fkt(det_center, last_hit):
    new_vec = last_hit - det_center
    distance = np.sqrt(new_vec[0]**2 + new_vec[1]**2 + new_vec[2]**2)
    return distance


@njit(parallel=True)
def main_ray_iteration_fkt(N_bounces, N_rays, max_parameter_t, ray_origin_matrix, ray_direction_matrix, 
                             det_center, det_radius, Object_lenght_vector, distance_vec, 
                             Triangle_matrix, Triangle_normal_matrix, 
                             Plane_point_matrix, Plane_normal_matrix,
                             Sphere_center_matrix, Sphere_radius):
    distance_vec_final = np.zeros(N_rays)
    for m in prange(N_rays):
        ray_origin = ray_origin_matrix[m,:]
        ray_direction = ray_direction_matrix[m,:]
        last_hit = ray_origin
        distance = distance_vec[m]
        for n in range(N_bounces):
            parameter_object = closest_object_fkt(Triangle_matrix, Triangle_normal_matrix, 
                                    Plane_point_matrix, Plane_normal_matrix,
                                    Sphere_center_matrix, Sphere_radius, Object_lenght_vector,
                                    det_center, det_radius, # load geometry
                                    max_parameter_t, ray_origin, ray_direction)
            # parameter_object = closest_object_tri_fkt(Triangle_matrix, Triangle_normal_matrix, 
            #                         Plane_point_matrix, Plane_normal_matrix,
            #                         Sphere_center_matrix, Sphere_radius, Object_lenght_vector,
            #                         det_center, det_radius, # load geometry
            #                         max_parameter_t, ray_origin, ray_direction)
            if parameter_object[0] == False:
                break
            t_parameter = parameter_object[0]
            suface_normal = parameter_object[1]
            ray_origin = intercetion_point_fkt(ray_origin, ray_direction, t_parameter)
            det_check = detector_check_fkt(ray_origin, det_center, det_radius)
            if det_check == True:
                distance_vec_final[m] = detector_lenght_cal_fkt(det_center, last_hit) + distance
                break
            distance = distance + t_parameter
            ray_direction = plane_tri_ref_fkt(ray_direction, suface_normal)
            last_hit = ray_origin
            
    return distance_vec_final


@njit
def remove_zeros_fkt(distance_vec_final):
    distance_vec_final = distance_vec_final[distance_vec_final != 0]
    return distance_vec_final


    
    




