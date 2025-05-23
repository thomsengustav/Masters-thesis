# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:16:59 2024

@author: thoms
"""

### signal_calc_functions

import numpy as np
from numba import njit

@njit
def func_rect(t, T):
    if abs(t) > T/2:
        return 0
    else:
        return 1
   
@njit
def func_chirp(t, tau, f_center, Chirp_duration, Chirp_rate):
    return func_rect(t-tau, Chirp_duration) * np.exp(1j*2*np.pi*(f_center*(t-tau) + 0.5*Chirp_rate*(t-tau)**2))

def find_azi_number_fkt(full_name):
    sub1 = "azi_"
    sub2 = "_start_"
    idx1 = full_name.find(sub1)
    idx2 = full_name.find(sub2)
     
    res = int(full_name[idx1 + len(sub1): idx2])
    return res

def find_start_end_fkt(full_name, name):
    start1 = "_start_"
    start2 = "_end_"
    idx1 = full_name.find(start1)
    idx2 = full_name.find(start2)
    start_azi = int(full_name[idx1 + len(start1): idx2])
    
    end1 = "_end_"
    end2 = "_" + name
    idx1 = full_name.find(end1)
    idx2 = full_name.find(end2)
    end_azi = int(full_name[idx1 + len(end1): idx2])
    return start_azi, end_azi