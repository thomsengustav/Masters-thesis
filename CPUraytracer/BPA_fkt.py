# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 11:31:25 2025

@author: thoms
"""

'''
BPA function
'''

import numpy as np
from scipy import signal
from Billede_danner_GPU_bistatic import Billede_danner_GPU_bistatic

# functions for running the BPA on simulation data from simulation name
def BPA_fkt(name, slant_ang, radar_range, num, x_min, x_max, y_min, y_max):  # ex. rast_name
    # get data
    azi_name = name + '_azimuth_vec.npy'
    mat_name = name + '_Matrix.npy'
    range_name = name + '_range_vec.npy'
    
    # constants radar settings
    T=5*10**-8#duration af puls
    f0=10*10**9#center freq
    B_c=4*10**9#Bandwidth
    K=B_c/T#chirp-rate, hvis det er chirp
    c=2.99792458*10**8
    k=2*np.pi*f0/c
    
    # range profile setup
    range_prof=np.load(mat_name)
    range_prof=np.transpose(range_prof)
    azi_v=np.load(azi_name)*np.pi/180
    sidste_t_vec=np.load(range_name)*2/c 
    
    R_Mat=np.zeros((azi_v.size,3))
    slant=slant_ang*np.pi/180
    R_Mat[:,2]=np.sin(slant)*radar_range
    R_Mat[:,0]=np.cos(slant)*np.sin(azi_v)*radar_range
    R_Mat[:,1]=np.cos(slant)*np.cos(azi_v)*radar_range
    
    # perform range compression
    dt=sidste_t_vec[1]-sidste_t_vec[0]
    tid_enkel=np.linspace(-0.6*T, 0.6*T, num=int(1.2*T/dt))
    range_ref=np.zeros(tid_enkel.size, dtype=np.complex_)
    for i in range(0,tid_enkel.size):
        t=tid_enkel[i]
        x1=(T)/2-((t-0)**2)**0.5
        range_ref[i]=np.heaviside(x1, 1)*np.exp(1j*(np.pi*K*(t-0)**2))
    
    t2_vec=np.linspace(tid_enkel[0]+sidste_t_vec[0], tid_enkel[-1]+sidste_t_vec[-1], num=tid_enkel.size+sidste_t_vec.size-1)
    range_compressed=np.zeros((R_Mat[:,0].size, t2_vec.size), dtype=np.complex_)
    for i in range(0,range_compressed[:,0].size):
        range_compressed[i,:]=signal.fftconvolve(range_prof[i,:], np.conjugate(range_ref))*dt
    
    # define SAR scene size and steps
    y_vec=np.linspace(x_min, x_max, num) #Scene center er sat i (x=0,y=0)
    x_vec=np.linspace(y_min, y_max, num) #x og y interval i meter, num=antal pixels i x og y dim
    I=np.zeros([x_vec.size,y_vec.size], dtype=np.complex_)#matrice med pixel værdier

    #position af emitter og detector, her er det sat til monostatic
    pos_emitter=np.transpose(R_Mat)
    pos_detector=pos_emitter
    range_compressed=np.transpose(range_compressed)
    d_dist=dt*c/2
    nul1=-int(t2_vec[0]*c/2/d_dist)# bruges til vælge rigtige indgange i range_compressed matricen i billede danner 

    I=Billede_danner_GPU_bistatic(y_vec,x_vec, pos_emitter, pos_detector, nul1, d_dist, k, range_compressed)
    return I, x_vec, y_vec

# functions for running the BPA on simulation data from simulation matrices 
def BPA_fkt2(name_azi, name_mat, name_range, slant_ang, radar_range, num, x_min, x_max, y_min, y_max):  # ex. rast_name
    # constants radar settings
    T=5*10**-8#duration af puls
    f0=10*10**9#center freq
    B_c=4*10**9#Bandwidth
    K=B_c/T#chirp-rate, hvis det er chirp
    c=2.99792458*10**8
    k=2*np.pi*f0/c
    
    range_prof=np.load(name_mat)
    range_prof=np.transpose(range_prof)
    azi_v=np.load(name_azi)*np.pi/180
    sidste_t_vec=np.load(name_range)*2/c 
    
    R_Mat=np.zeros((azi_v.size,3))
    slant=slant_ang*np.pi/180
    R_Mat[:,2]=np.sin(slant)*radar_range
    R_Mat[:,0]=np.cos(slant)*np.sin(azi_v)*radar_range
    R_Mat[:,1]=np.cos(slant)*np.cos(azi_v)*radar_range
    
    # perform range compression
    dt=sidste_t_vec[1]-sidste_t_vec[0]
    tid_enkel=np.linspace(-0.6*T, 0.6*T, num=int(1.2*T/dt))
    range_ref=np.zeros(tid_enkel.size, dtype=np.complex_)
    for i in range(0,tid_enkel.size):
        t=tid_enkel[i]
        x1=(T)/2-((t-0)**2)**0.5
        range_ref[i]=np.heaviside(x1, 1)*np.exp(1j*(np.pi*K*(t-0)**2))
    
    t2_vec=np.linspace(tid_enkel[0]+sidste_t_vec[0], tid_enkel[-1]+sidste_t_vec[-1], num=tid_enkel.size+sidste_t_vec.size-1)
    range_compressed=np.zeros((R_Mat[:,0].size, t2_vec.size), dtype=np.complex_)
    for i in range(0,range_compressed[:,0].size):
        range_compressed[i,:]=signal.fftconvolve(range_prof[i,:], np.conjugate(range_ref))*dt

    y_vec=np.linspace(x_min, x_max, num) #Scene center er sat i (x=0,y=0)
    x_vec=np.linspace(y_min, y_max, num) #x og y interval i meter, num=antal pixels i x og y dim
    I=np.zeros([x_vec.size,y_vec.size], dtype=np.complex_)#matrice med pixel værdier

    #position af emitter og detector, her er det sat til monostatic
    pos_emitter=np.transpose(R_Mat)
    pos_detector=pos_emitter
    range_compressed=np.transpose(range_compressed)
    d_dist=dt*c/2
    nul1=-int(t2_vec[0]*c/2/d_dist)# bruges til vælge rigtige indgange i range_compressed matricen i billede danner 

    I=Billede_danner_GPU_bistatic(y_vec,x_vec, pos_emitter, pos_detector, nul1, d_dist, k, range_compressed)
    return I, x_vec, y_vec