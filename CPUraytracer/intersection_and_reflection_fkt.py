# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 13:38:54 2024

@author: thoms
"""

#### calculate direction and intersection of perfectly reflected ray 

'''
Ray tracing functions
'''

import numpy as np

#### numba test

from numba import njit, float64
import numpy as np
from fast_vec_operators import dot_fkt, norm_vec_fkt


@njit
def intercetion_point_fkt(ray_origin, ray_direction, t_parameter):
    return ray_origin + t_parameter * ray_direction


@njit
def normal_direction_check_fkt(ray_direction, object_normal):
    ray_surface_dot = dot_fkt(ray_direction, object_normal)
    if ray_surface_dot < 0:
        return ray_surface_dot, object_normal
    else:
        return - ray_surface_dot, - object_normal


@njit
def plane_tri_ref_fkt(ray_direction, object_normal):
    ray_normal_dot, new_object_normal = normal_direction_check_fkt(ray_direction, object_normal)
    ref_direction = ray_direction - 2 * ray_normal_dot * new_object_normal
    return ref_direction

@njit
def sphere_ref_fkt(ray_direction, object_normal):
    ref_direction = ray_direction - 2 * dot_fkt(ray_direction, object_normal) * object_normal
    return ref_direction

#### test numba
@njit
def sphere_surface_norm_fkt(ray_origin, ray_direction, sphere_center, radius, t_parameter):
    intersection_point = ray_origin + ray_direction * t_parameter
    surface_vec = intersection_point - sphere_center
    norm_surface_vec = norm_vec_fkt(surface_vec)
    return norm_surface_vec
