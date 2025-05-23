# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 14:21:17 2025

@author: thoms
"""

import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import fnmatch
import os

from scipy import signal


def get_puls():
    delta_d = 0.001 # in compressed land
    max_d = 320 # in compressed land
    
    T=5*10**-8
    f0=10*10**9
    B_c=4*10**9
    K=B_c/T
    
    c=299792458.0
    dt= delta_d / c
    tn=0
    tid_enkel=np.linspace(-0.6*T, 0.6*T, num=int(1.2*T/dt))
    enkel_puls=np.zeros(tid_enkel.size, dtype=np.complex_)
    for i in range(0,tid_enkel.size):
        t=tid_enkel[i]
        x1=(T)/2-((t-tn)**2)**0.5
        enkel_puls[i]=np.heaviside(x1, 1)*np.exp(1j*(2*np.pi*f0*(t-tn)+np.pi*K*(t-tn)**2))
    indgange = int(max_d / delta_d)
    t1_vec=np.linspace(0, dt*(indgange-1), num=indgange)
    modulator=np.zeros(t1_vec.size, dtype=np.complex_)
    modulator=np.exp(-1j*2*np.pi*f0*t1_vec)
    return enkel_puls, modulator, indgange
    
def convolution_spec_calc(sparse_vec, sparse_his, enkel_puls, modulator, indgange):
    delta_d = 0.001 # in compressed land
    max_d = 320 # in compressed land
    range_vec = np.zeros(indgange)
    for i in range(int(len(sparse_vec))):
        range_val = sparse_vec[i]
        sparse_index = int(range_val / delta_d)
        range_vec[sparse_index] = sparse_his[i]
        
    rangeM_signal=np.zeros(indgange, dtype=np.complex_)
    rangeM_signal=signal.fftconvolve(range_vec, enkel_puls,'same')*modulator
    
    total_spec_R = np.linspace(0, max_d / 2, indgange)
    
    return rangeM_signal, total_spec_R

from signal_calc_functions import find_azi_number_fkt, find_start_end_fkt

def SAR_signal_fkt(name, center_frequency, Bandwidth, chirp_duration):
    
    names = fnmatch.filter(os.listdir('C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\'), 'rast_azi_*' + name + '*.txt')
    azi_steps = len(names)

    azi_start, azi_end = find_start_end_fkt(names[0],name)
    azimuthal_vec =  np.linspace(azi_start, azi_end, azi_steps)

    data = np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\" + names[0], delimiter=',')
    distance_vec = data[:,0]
    weight_vec = data[:,1]
    
    enkel_puls, modulator, indgange = get_puls()
    
    signal_out = convolution_spec_calc(distance_vec, weight_vec, enkel_puls, modulator, indgange)
    DATA_matrix = np.zeros((len(signal_out[1]),azi_steps), dtype=np.complex_)
    range_vec = signal_out[1]

    for i in range(azi_steps):
        print(i)
        azi_num = find_azi_number_fkt(names[i])
        
        data = np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\" + names[i], delimiter=',')
        distance_vec = data[:,0]
        weight_vec = data[:,1]
        signal_out = convolution_spec_calc(distance_vec, weight_vec, enkel_puls, modulator, indgange)
        DATA_matrix[:,int(azi_num-1)] = signal_out[0]
    
    np.save("rast_" + name + "_Matrix", DATA_matrix)
    np.save("rast_" + name + "_range_vec", range_vec)
    np.save("rast_" + name + "_azimuth_vec", azimuthal_vec)
    
    return DATA_matrix, range_vec, azimuthal_vec

if __name__ == '__main__':    
    test, test1, test2 = SAR_signal_fkt("box_clutter_3", 10e9, 4e9, 5e-8)
