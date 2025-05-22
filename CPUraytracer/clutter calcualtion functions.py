# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 09:52:13 2024

@author: thoms
"""

'''
clutter calcualtion functions
'''

import numpy as np
from numba import njit

@njit
def sigma_0_fkt(theta, P_vec): # calculate terrain reflectivity from statistical dist. theta angle of incidence
    theta_rad = theta*np.pi/180
    sigma_0 = P_vec[0] + P_vec[1]*np.exp(-P_vec[2]*theta_rad) + P_vec[3]*np.cos(P_vec[4]*theta_rad + P_vec[5]) # theta in rad
    return sigma_0 
    # get P1-P6 from Handbook of radar scattering statistics for terrain
    
@njit
def get_normal_var(sigma_0, dr, dthe, slant_ang): # calculate variance for getting reflectivity
    slant_rad = slant_ang*np.pi/180
    var_2 = (sigma_0*dr*dthe)/(2*np.cos(slant_rad)) # angles in rad
    return var_2

@njit
def get_complex_ref(variance_2, N): # generate complex reflectivity
    real = np.random.normal(0, variance_2, N)
    imag = np.random.normal(0, variance_2, N)
    complex_ref_vec = np.zeros(N, dtype=np.complex128) 
    complex_ref_vec[:] = real + 1j*imag
    return complex_ref_vec

# clutter_data = (dstep, slant_angle, P_vec)

@njit
def get_N_complex_ref(clutter_data, N_set):
    dstep = clutter_data[0]
    slant_ang = clutter_data[1]
    P_vec = clutter_data[2]
    sigma_0 = sigma_0_fkt(slant_ang, P_vec)
    variance_2 = get_normal_var(sigma_0, dstep, dstep, slant_ang)
    R_vec_complex = get_complex_ref(variance_2, N_set)
    return R_vec_complex
    