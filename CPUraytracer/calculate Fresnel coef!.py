# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 15:45:28 2024

@author: thoms
"""

'''
calculate Fresnel coef!
testing of ray-tracing polarization
'''


# input: theta, n, k
# output: R

from numba import njit
import numpy as np

def get_theta_fkt(tri_norm, ray_dir):
    theta = np.arccos(np.dot(-tri_norm, ray_dir))
    return theta

def simple_power_R_fkt(theta, n, k):
    n_squared = n*n
    k_squared = k*k    
    sin_theta = np.sin(theta)
    sin_squared = sin_theta * sin_theta
    n_k_sin_term = n_squared - k_squared - sin_squared
    n_k_4 = 4 * n_squared * k_squared
    n_k_sin_root = np.sqrt(n_k_sin_term + n_k_4)
    
    a_squared = (n_k_sin_root + n_k_sin_term)*0.5
    b_squared = (n_k_sin_root - n_k_sin_term)*0.5
    a2 = 2 * np.sqrt(a_squared) 
    b2 = 2 * np.sqrt(b_squared) # need for calculating phase shift later
    ab_squared_sum = a_squared + b_squared
    
    cos_theta = np.cos(theta)
    cos_squared = cos_theta * cos_theta
    tan_theta = np.tan(theta)
    tan_squared = tan_theta * tan_theta
    
    term_per1 = ab_squared_sum + cos_squared
    term_per2 = a2 * cos_theta
    div_per = term_per1 + term_per2
    F_per = (term_per1 - term_per2) / (div_per) # perpendicular to the plane of incident
    
    term_par1 = ab_squared_sum + sin_squared * tan_squared
    term_par2 = a2 * sin_theta * tan_theta
    div_par = term_par1 + term_par2
    F_par = (term_par1 - term_par2) / (div_par)  # parallel
    
    # simple avg for ref power
    F = (F_par + F_per)*0.5
    return F

def power_R_and_phase_fkt(theta, n, k):
    n_squared = n*n
    k_squared = k*k    
    sin_theta = np.sin(theta)
    sin_squared = sin_theta * sin_theta
    n_k_sin_term = n_squared - k_squared - sin_squared
    n_k_4 = 4 * n_squared * k_squared
    n_k_sin_root = np.sqrt(n_k_sin_term + n_k_4)
    
    a_squared = (n_k_sin_root + n_k_sin_term)*0.5
    b_squared = (n_k_sin_root - n_k_sin_term)*0.5
    a2 = 2 * np.sqrt(a_squared) 
    b2 = 2 * np.sqrt(b_squared) # need for calculating phase shift later
    ab_squared_sum = a_squared + b_squared
    
    cos_theta = np.cos(theta)
    cos_squared = cos_theta * cos_theta
    tan_theta = np.tan(theta)
    tan_squared = tan_theta * tan_theta
    
    term_per1 = ab_squared_sum + cos_squared
    term_per2 = a2 * cos_theta
    div_per = term_per1 + term_per2
    F_per = (term_per1 - term_per2) / (div_per) # perpendicular to the plane of incident
    
    term_par1 = ab_squared_sum + sin_squared * tan_squared
    term_par2 = a2 * sin_theta * tan_theta
    div_par = term_par1 + term_par2
    F_par = (term_par1 - term_par2) / (div_par)  # parallel
    
    phase_per_numerator = b2 * cos_theta
    phase_per_denumerator = cos_squared - ab_squared_sum
    phase_per = np.arctan(phase_per_numerator / phase_per_denumerator)
    
    phase_par_numerator = cos_theta * (b2 * (n_k_sin_term + sin_squared) - 2 * a2 * n * k) 
    phase_par_denumerator = (n_squared + k_squared)**2 * cos_squared - ab_squared_sum
    phase_par = np.arctan(phase_par_numerator / phase_par_denumerator)
    return F_per, F_par, phase_per, phase_par
    
    
    