# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 19:10:32 2025

@author: thoms
"""

'''
functions for calculating the radar signal from polRT output files
'''

import numpy as np
import fnmatch
import os

from signal_from_txt_rast_sparse import get_puls, convolution_spec_calc
from signal_calc_functions import find_azi_number_fkt, find_start_end_fkt

def SAR_signal_HV_fkt(name):
    
    names = fnmatch.filter(os.listdir('C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\'), 'HVRT_azi_*' + name + '*.txt')
    azi_steps = len(names)

    azi_start, azi_end = find_start_end_fkt(names[0],name)
    azimuthal_vec =  np.linspace(azi_start, azi_end, azi_steps)

    data = np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\" + names[0], delimiter=',')
    distance_vec = data[:,0]
    weight_vec = data[:,1]
    
    enkel_puls, modulator, indgange = get_puls()
    
    signal_out = convolution_spec_calc(distance_vec, weight_vec, enkel_puls, modulator, indgange)
    DATA_matrixHH = np.zeros((len(signal_out[1]),azi_steps), dtype=np.complex_)
    DATA_matrixHV = np.zeros((len(signal_out[1]),azi_steps), dtype=np.complex_)
    DATA_matrixVH = np.zeros((len(signal_out[1]),azi_steps), dtype=np.complex_)
    DATA_matrixVV = np.zeros((len(signal_out[1]),azi_steps), dtype=np.complex_)
    range_vec = signal_out[1]

    for i in range(azi_steps):
        print(i)
        azi_num = find_azi_number_fkt(names[i])
        
        data = np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\" + names[i], delimiter=',')
        distance_vec = data[:,0]
        HH = data[:,1]
        HV = data[:,2]
        VH = data[:,3]
        VV = data[:,4]
        signal_HH = convolution_spec_calc(distance_vec, HH, enkel_puls, modulator, indgange)
        signal_HV = convolution_spec_calc(distance_vec, HV, enkel_puls, modulator, indgange)
        signal_VH = convolution_spec_calc(distance_vec, VH, enkel_puls, modulator, indgange)
        signal_VV = convolution_spec_calc(distance_vec, VV, enkel_puls, modulator, indgange)
        DATA_matrixHH[:,int(azi_num-1)] = signal_HH[0]
        DATA_matrixHV[:,int(azi_num-1)] = signal_HV[0]
        DATA_matrixVH[:,int(azi_num-1)] = signal_VH[0]
        DATA_matrixVV[:,int(azi_num-1)] = signal_VV[0]
    
    np.save("HVRT_" + name + "_MatrixHH", DATA_matrixHH)
    np.save("HVRT_" + name + "_MatrixHV", DATA_matrixHV)
    np.save("HVRT_" + name + "_MatrixVH", DATA_matrixVH)
    np.save("HVRT_" + name + "_MatrixVV", DATA_matrixVV)
    np.save("HVRT_" + name + "_range_vec", range_vec)
    np.save("HVRT_" + name + "_azimuth_vec", azimuthal_vec)
    
    return


def SAR_signal_fkt(name, center_frequency, Bandwidth, chirp_duration):
    
    names = fnmatch.filter(os.listdir('C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\'), 'PolRT_azi_*' + name + '*.txt')
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
    
    np.save("PolRT_" + name + "_Matrix", DATA_matrix)
    np.save("PolRT_" + name + "_range_vec", range_vec)
    np.save("PolRT_" + name + "_azimuth_vec", azimuthal_vec)
    
    return DATA_matrix, range_vec, azimuthal_vec

if __name__ == '__main__':     
    test, test1, test2 = SAR_signal_fkt("pol_box_new_test_1", 10e9, 4e9, 5e-8)
