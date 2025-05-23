# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 21:43:56 2025

@author: thoms
"""

'''
new save method for smaller data_sets!

Use range buckets.
For each ray, calculate range bucket and add weight
'''

import numpy as np
from numba import njit

@njit
def get_range_histogram(distance_vec, weight_vec, t_max, resolution):
    max_dist = t_max * 2
    N_his = int(max_dist / resolution) + 1
    range_vec = np.linspace(0,int(max_dist), N_his)
    range_his = np.zeros(N_his)
    
    N_dist = int(len(distance_vec))
    resolition_inv = 1 / resolution
    range_index_vec = distance_vec * resolition_inv
    for i in range(N_dist):
        range_his[int(range_index_vec[i])] += weight_vec[i]

    return range_vec, range_his

@njit
def get_range_histogram_HV(distance_vec, HH, HV, VH, VV, t_max, resolution):
    max_dist = t_max * 2
    N_his = int(max_dist / resolution) + 1
    range_vec = np.linspace(0,int(max_dist), N_his)
    his_HH = np.zeros(N_his)
    his_HV = np.zeros(N_his)
    his_VH = np.zeros(N_his)
    his_VV = np.zeros(N_his)
    
    N_dist = int(len(distance_vec))
    resolition_inv = 1 / resolution
    range_index_vec = distance_vec * resolition_inv
    for i in range(N_dist):
        his_HH[int(range_index_vec[i])] += HH[i]
        his_HV[int(range_index_vec[i])] += HV[i]
        his_VH[int(range_index_vec[i])] += VH[i]
        his_VV[int(range_index_vec[i])] += VV[i]

    return range_vec, his_HH, his_HV, his_VH, his_VV

@njit
def remove_zeros_range_HV(range_vec, his_HH, his_HV, his_VH, his_VV):
    N_his = int(len(range_vec))
    new_his_HH = np.zeros(N_his)
    new_his_HV = np.zeros(N_his)
    new_his_VH = np.zeros(N_his)
    new_his_VV = np.zeros(N_his)
    his_prod = his_HH*his_HV*his_VH*his_VV
    new_range_vec = np.zeros(N_his)
    non_zero_counter = 0
    for i in range(N_his):
        his_val = his_prod[i]
        if his_val != 0:
            new_his_HH[non_zero_counter] = his_HH[i]
            new_his_HV[non_zero_counter] = his_HV[i]
            new_his_VH[non_zero_counter] = his_VH[i]
            new_his_VV[non_zero_counter] = his_VV[i]
            new_range_vec[non_zero_counter] = range_vec[i] 
            non_zero_counter += 1
    fin_range_vec = new_range_vec[:non_zero_counter]
    fin_HH = new_his_HH[:non_zero_counter]
    fin_HV = new_his_HV[:non_zero_counter]
    fin_VH = new_his_VH[:non_zero_counter]
    fin_VV = new_his_VV[:non_zero_counter]
    return fin_range_vec, fin_HH, fin_HV, fin_VH, fin_VV

@njit
def remove_zeros_range(range_vec, range_his):
    N_his = int(len(range_vec))
    new_range_his = np.zeros(N_his)
    new_range_vec = np.zeros(N_his)
    non_zero_counter = 0
    for i in range(N_his):
        his_val = range_his[i]
        if his_val != 0:
            new_range_his[non_zero_counter] = his_val
            new_range_vec[non_zero_counter] = range_vec[i] 
            non_zero_counter += 1
    fin_range_vec = new_range_vec[:non_zero_counter]
    fin_range_his = new_range_his[:non_zero_counter]
    return fin_range_vec, fin_range_his
