# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 12:21:37 2024

@author: thoms
"""

#### calculate return signal from measured distances.

import numpy as np
from numba import njit
import pandas as pd
import matplotlib.pylab as plt
import fnmatch
import os



# calculate chirp signal from t and tau (calculated from distance of ray)

from signal_calc_functions import func_chirp, func_rect, find_azi_number_fkt, find_start_end_fkt

def signal_from_distance_fkt(total_distance_vec, center_frequency, Bandwidth, chirp_duration):
    f_center = center_frequency # 10e9 # Hz
    Bandwidth = Bandwidth # 4e9 # Hz
    Chirp_duration = chirp_duration # 5e-8 # s
    Chirp_rate = Bandwidth / Chirp_duration # Hz/s
    
    # time setup
    puls_time_steps = 18000
    total_time_high = 12e-7
    puls_time_low = -0.6*Chirp_duration
    puls_time_high = 0.6*Chirp_duration
    delta_t = (puls_time_high-puls_time_low)/puls_time_steps
    time_steps = int((total_time_high - puls_time_low)/delta_t)
    
    c_light = 299792458 # m / s
    time_collected_rays = total_distance_vec / c_light

    time_puls_vec = np.linspace(puls_time_low, puls_time_high, puls_time_steps)
    spec_puls_vec = np.zeros(len(time_puls_vec), dtype=np.complex_)
    counter=0
    for t in time_puls_vec:
        spec_puls_vec[counter] = func_chirp(t, 0, f_center, Chirp_duration, Chirp_rate)
        counter +=1

    total_time_vec = np.linspace(puls_time_low, total_time_high,int(time_steps))
    demodulated_vec_mix = np.exp(-1j*2*np.pi*f_center*total_time_vec)
    total_spec = np.zeros(int(time_steps), dtype=np.complex_)
    
    for n in range(0, len(time_collected_rays)):
        displacement_time = time_collected_rays[n]
        spec_index = int((displacement_time/delta_t))
        total_spec[spec_index : int(spec_index + puls_time_steps)] += spec_puls_vec
    
    total_spec_R = total_time_vec*c_light/2
    # plt.plot(total_spec_R, total_spec)
    # plt.xlim(94,105)
    # plt.title('raw signal')
    # plt.xlabel("distance (m)")
    # plt.show()
    
    total_spec_demod = total_spec * demodulated_vec_mix
    
    # plt.plot(total_spec_R, total_spec_demod)
    # plt.xlim(94,105)
    # plt.title('Demodulated signal')
    # plt.xlabel("distance (m)")
    # plt.show()
    
    return total_spec_demod, total_spec_R




def SAR_signal_fkt(name, center_frequency, Bandwidth, chirp_duration):
    
    names = fnmatch.filter(os.listdir('C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\'), 'azi_*' + name + '*.txt')
    azi_steps = len(names)
    
    azi_start, azi_end = find_start_end_fkt(names[0],name)
    azimuthal_vec =  np.linspace(azi_start, azi_end, azi_steps)
    
    distance_data_pd = pd.read_table("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\" + names[0])
    distance_vec = distance_data_pd.to_numpy()
    
    signal_out = signal_from_distance_fkt(distance_vec, center_frequency, Bandwidth, chirp_duration)
    DATA_matrix = np.zeros((len(signal_out[1]),azi_steps), dtype=np.complex_)
    range_vec = signal_out[1]
    
    for i in range(azi_steps):
        azi_num = find_azi_number_fkt(names[i])
        
        distance_data_pd = pd.read_table("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\" + names[i])
        distance_vec = distance_data_pd.to_numpy()
        
        signal_out = signal_from_distance_fkt(distance_vec, center_frequency, Bandwidth, chirp_duration)
        DATA_matrix[:,int(azi_num-1)] = signal_out[0]
    
    np.save(name + "_Matrix", DATA_matrix)
    np.save(name + "_range_vec", range_vec)
    np.save(name + "_azimuth_vec", azimuthal_vec)
    
    return DATA_matrix, range_vec, azimuthal_vec
    
SAR_signal_fkt("sphere4_smooth", 10e9, 4e9, 5e-8)


# plt.plot(range_vec, DATA_matrix[:,1])
# plt.xlim(94,105)
# plt.title('Demodulated signal')
# plt.xlabel("distance (m)")
# plt.show()


# plt.pcolormesh(azimuthal_vec, range_vec, abs(DATA_matrix))
# plt.xlim(-5, 5)
# plt.ylim(95, 105)
# plt.xlabel("Azimuthal angle (°)")
# plt.ylabel("Range (m)")
# plt.title('amplitude')
# plt.colorbar()
# plt.show()