# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 09:20:44 2024

@author: thoms
"""

#### Calculate rotation matrix

"""
Inputs: Radar_location (np.array [x,y,z])

Output: 3x3 rotation matrix

Assumptions: Scene center always located at [0,0,0]
"""
from numba import njit
import numpy as np
from fast_vec_operators import norm_vec_fkt

@njit
def beam_direction_fkt(Radar_location):
    beam_direction = - norm_vec_fkt(Radar_location)
    return beam_direction

@njit
def z_angle_fkt(beam_direction):
    plane_vec = beam_direction - np.array([0.,0.,1.]) * beam_direction[2]
    plane_vec_norm = plane_vec / np.linalg.norm(plane_vec)
    x_norm = np.array([1.,0.,0.])
    alfa = np.arccos(np.dot(x_norm, plane_vec_norm))
    if plane_vec_norm[1] > 0:
        return alfa
    else:
        alfa = 2*np.pi - alfa
        return alfa

@njit
def rotation_matrix_fkt(beam_direction): # keep non optimized, uses unsupported np functions
    y_angle = np.arccos(beam_direction[2])
    Ry = np.array([[np.cos(y_angle), 0., np.sin(y_angle)],[0., 1., 0.], [-np.sin(y_angle), 0., np.cos(y_angle)]])
    
    z_angle = z_angle_fkt(beam_direction)
    Rz = np.array([[np.cos(z_angle), -np.sin(z_angle), 0.], [np.sin(z_angle), np.cos(z_angle), 0.], [0., 0., 1.]])
    
    ### matmul
    new_rot = np.zeros((3,3))
    new_rot[0,0] = Rz[0,0]*Ry[0,0] + Rz[0,1]*Ry[1,0] + Rz[0,2]*Ry[2,0]
    new_rot[1,0] = Rz[1,0]*Ry[0,0] + Rz[1,1]*Ry[1,0] + Rz[1,2]*Ry[2,0]
    new_rot[2,0] = Rz[2,0]*Ry[0,0] + Rz[2,1]*Ry[1,0] + Rz[2,2]*Ry[2,0]

    new_rot[0,1] = Rz[0,0]*Ry[0,1] + Rz[0,1]*Ry[1,1] + Rz[0,2]*Ry[2,1]
    new_rot[1,1] = Rz[1,0]*Ry[0,1] + Rz[1,1]*Ry[1,1] + Rz[1,2]*Ry[2,1]
    new_rot[2,1] = Rz[2,0]*Ry[0,1] + Rz[2,1]*Ry[1,1] + Rz[2,2]*Ry[2,1]

    new_rot[0,2] = Rz[0,0]*Ry[0,2] + Rz[0,1]*Ry[1,2] + Rz[0,2]*Ry[2,2]
    new_rot[1,2] = Rz[1,0]*Ry[0,2] + Rz[1,1]*Ry[1,2] + Rz[1,2]*Ry[2,2]
    new_rot[2,2] = Rz[2,0]*Ry[0,2] + Rz[2,1]*Ry[1,2] + Rz[2,2]*Ry[2,2]
    
    return new_rot
