# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 12:36:57 2024

@author: thoms
"""

### calculate radar signal from rasterization!

import numpy as np
from numba import njit, prange
import fnmatch
import os
import time
import matplotlib.pyplot as plt

from signal_calc_functions import func_chirp, func_rect, find_azi_number_fkt, find_start_end_fkt

@njit
def make_final_sinal(time_collected_rays, delta_t, puls_time_steps, time_steps, spec_puls_vec, weight_vec):
    total_spec = np.zeros(int(time_steps), dtype=np.complex_)
    for n in prange(int(len(time_collected_rays))):
        displacement_time = time_collected_rays[n]
        spec_index = int((displacement_time / delta_t))
        total_spec[spec_index : int(spec_index + puls_time_steps)] += spec_puls_vec * weight_vec[n]
    return total_spec


def signal_from_distance_weighted_fkt(distance_vec, weight_vec): #, center_frequency, Bandwidth, chirp_duration):
    f_center = 10e9 # Hz
    Bandwidth = 4e9 # Hz
    Chirp_duration =  5e-8 # s
    Chirp_rate = Bandwidth / Chirp_duration # Hz/s
    
    # time setup
    puls_time_steps = 17989
    total_time_high = 12e-7
    puls_time_low = -0.6*Chirp_duration
    puls_time_high = 0.6*Chirp_duration
    delta_t = (puls_time_high-puls_time_low)/puls_time_steps
    time_steps = int((total_time_high - puls_time_low)/delta_t)
    
    c_light = 299792458.0 # m / s
    time_collected_rays = distance_vec * (1 / c_light)

    time_puls_vec = np.linspace(puls_time_low, puls_time_high, puls_time_steps)
    spec_puls_vec = np.zeros(len(time_puls_vec), dtype=np.complex_)
    counter=0
    for t in time_puls_vec:
        spec_puls_vec[counter] = func_chirp(t, 0, f_center, Chirp_duration, Chirp_rate)
        counter +=1
    
    total_time_vec = np.linspace(puls_time_low, total_time_high,int(time_steps))
    demodulated_vec_mix = np.exp(-1j*2*np.pi*f_center*total_time_vec)
    total_spec = np.zeros(int(time_steps), dtype=np.complex_)
    
    t0 = time.time()
    for n in range(0, len(time_collected_rays)):
        displacement_time = time_collected_rays[n]
        spec_index = int((displacement_time/delta_t))
        total_spec[spec_index : int(spec_index + puls_time_steps)] += spec_puls_vec * weight_vec[n]
    # print(time.time()-t0)
        
    total_spec_R = total_time_vec*c_light/2
    
    total_spec_demod = total_spec * demodulated_vec_mix
    
    return total_spec_demod, total_spec_R

from scipy import signal

def convolution_spec_calc(distance_vec, weight_vec):
    delta_d = 0.001
    max_d = 360
    #chirp signal
    T=5*10**-8
    f0=10*10**9
    B_c=4*10**9
    K=B_c/T
    
    c=299792458.0
    k=2*np.pi*f0/c
    dt= delta_d / c
    tn=0
    tid_enkel=np.linspace(-0.6*T, 0.6*T, num=int(1.2*T/dt))
    enkel_puls=np.zeros(tid_enkel.size, dtype=np.complex_)
    for i in range(0,tid_enkel.size):
        t=tid_enkel[i]
        x1=(T)/2-((t-tn)**2)**0.5
        enkel_puls[i]=np.heaviside(x1, 1)*np.exp(1j*(2*np.pi*f0*(t-tn)+np.pi*K*(t-tn)**2))
    
    # fig, axa2 = plt.subplots()
    # axa2.set_title('enkel ikke moduleret')    
    # axa2.plot(tid_enkel,np.real(enkel_puls))
    #axa.set_xlim([-5*10**-8,-4.9*10**-8])
    
    indgange = int(max_d / delta_d)
    t1_vec=np.linspace(0, dt*(indgange-1), num=indgange)
    modulator=np.zeros(t1_vec.size, dtype=np.complex_)
    for i in range(0,t1_vec.size):
        modulator[i]=np.exp(-1j*2*np.pi*f0*t1_vec[i])
    
    range_vec = np.zeros(indgange)
    for i in range(int(len(distance_vec))):
        div = distance_vec[i] / delta_d
        index = np.floor(div)
        range_vec[int(index)] += 1 * weight_vec[i]
    
    # fig, axa2 = plt.subplots()
    # axa2.set_title('range_bin')    
    # axa2.plot(range_vec)
    
    rangeM_signal=np.zeros(indgange, dtype=np.complex_)
    rangeM_signal=signal.fftconvolve(range_vec, enkel_puls,'same')*modulator
    
    total_spec_R = np.linspace(0, max_d / 2, indgange)
    
    # fig, axa3 = plt.subplots()
    # axa3.set_title('conv')    
    # axa3.plot(total_spec_R, np.real(rangeM_signal))
    # axa3.set_xlim([60,100])  
    
    return rangeM_signal, total_spec_R

def SAR_signal_fkt(name, center_frequency, Bandwidth, chirp_duration):
    
    names = fnmatch.filter(os.listdir('C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\'), 'rast_azi_*' + name + '*.txt')
    azi_steps = len(names)

    azi_start, azi_end = find_start_end_fkt(names[0],name)
    azimuthal_vec =  np.linspace(azi_start, azi_end, azi_steps)

    data = np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\" + names[0], delimiter=',')
    distance_vec = data[:,0]
    weight_vec = data[:,1]
    
    signal_out = convolution_spec_calc(distance_vec, weight_vec)
    # signal_out = signal_from_distance_weighted_fkt(distance_vec, weight_vec)
    DATA_matrix = np.zeros((len(signal_out[1]),azi_steps), dtype=np.complex_)
    range_vec = signal_out[1]

    for i in range(azi_steps):
        print(i)
        azi_num = find_azi_number_fkt(names[i])
        
        data = np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\" + names[i], delimiter=',')
        distance_vec = data[:,0]
        weight_vec = data[:,1]
        signal_out = convolution_spec_calc(distance_vec, weight_vec)
        # signal_out = signal_from_distance_weighted_fkt(distance_vec, weight_vec)
        DATA_matrix[:,int(azi_num-1)] = signal_out[0]
    
    np.save("rast_" + name + "_Matrix", DATA_matrix)
    np.save("rast_" + name + "_range_vec", range_vec)
    np.save("rast_" + name + "_azimuth_vec", azimuthal_vec)
    
    return DATA_matrix, range_vec, azimuthal_vec
    
SAR_signal_fkt("box_rast_clutter_3", 10e9, 4e9, 5e-8)
