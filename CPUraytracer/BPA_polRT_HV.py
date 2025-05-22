# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 15:03:18 2025

@author: thoms
"""

'''
BPA on polarimetry simulations
'''
import numpy as np

#simulation name
name = 'råsted_test_new_rot_2'

# setup
slant_ang = 25
radar_range = 100
num=900
x_min = -5
x_max = 5
y_min = -5
y_max = 5

x_vec = np.linspace(x_min,x_max,num)
y_vec = np.linspace(y_min,y_max,num)


# plot settings
calculate_signal = False
calculate_SAR = False
plot_DB = False
multilook_var_SAR = True
plot_SAR = True

plot_polarimetry = True
plot_RGB = True
pol_dB = False
plot_SAR_final = False

multilook_var = True
C_diag = False
C_multi = False

Multi_dim = 5
cross_par = 0

import matplotlib.pyplot as plt

from signal_pol_RT import SAR_signal_HV_fkt
from BPA_fkt import BPA_fkt2
# from Freeman_decomp import get_C_from_im
import matplotlib.colors 


custom_map_linlog = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#000000", "#3F3F3F","#737373","#989898","#B3B3B3","#C8C8C8","#D8D8D8","#E6E6E6","#F3F3F3","#FFFFFF"])

azi_name = 'HVRT_' + name + '_azimuth_vec.npy'
range_name = 'HVRT_' + name + '_range_vec.npy'

pol = ['HH', 'HV', 'VH', 'VV']

I_mat = np.zeros((num,num,4),dtype = np.complex128)
I_mat_dB = np.zeros((num,num,4),dtype = np.complex128)

def multilook(SAR_image, N_dim):
    N = int(len(SAR_image[0,:]))
    M = int(len(SAR_image[:,0]))
    
    M_multi = int(np.floor(M/N_dim))
    N_multi = int(np.floor(N/N_dim))
    
    multilook_image = np.zeros((N_multi, M_multi))

    for n in range(N_multi):
        for m in range(M_multi):
            multilook_image[n,m] += np.sum(abs(SAR_image[n*N_dim:(n+1)*N_dim,m*N_dim:(m+1)*N_dim]))
            
    return multilook_image


# calculate and save signals
if calculate_signal == True:
    SAR_signal_HV_fkt(name)


if calculate_SAR == True:
    for i in range(4):
        name_mat = 'HVRT_' + name + '_matrix' + pol[i] + '.npy'
        I, x_vec, y_vec = BPA_fkt2(azi_name, name_mat, range_name, slant_ang, radar_range, num, x_min, x_max, y_min, y_max) 
        np.save('I_mat_' + name_mat, I)
        I_mat[:,:,i] = I
        
        if multilook_var_SAR == True:
            Iplot = multilook(I, Multi_dim)
            N_i = int(len(Iplot[0,:,]))
            x_vec = np.linspace(x_min,x_max,N_i)
            y_vec = np.linspace(y_min,y_max,N_i)
            
        fig, axa = plt.subplots()
        axa.set_title('Village' + '_' + pol[i])
        fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(Iplot[0:Iplot[:,1].size-1,0:Iplot[1,:].size-1])*100, cmap=custom_map_linlog)
        cbar=fig.colorbar(fpf,ax=axa)
        plt.xlabel("Range [m]", fontsize=16)
        plt.ylabel("Azimuth [m]", fontsize=16)
        axa.tick_params(labelsize=14)
        cbar.set_label("Intensity [a.u.]", fontsize=14)
        axa.set_aspect('equal')
        
        if plot_DB == True:
            I_DB=20*np.log10(np.abs(Iplot)/np.max(np.abs(Iplot)))
            # I_mat_dB[:,:,i] = I_DB
            fig, axa = plt.subplots()
            axa.set_title(name + '_' + pol[i] + '_dB')
            fpf=axa.pcolormesh(y_vec, x_vec, (I_DB[0:I_DB[:,1].size-1,0:I_DB[1,:].size-1]),cmap='gray',vmin=-25)#, vmax=60000)#, vmax=60000)#,vmin=0, vmax=500)
            cbar=fig.colorbar(fpf,ax=axa)
            #axa.set_xlim([-20,-15])
            # axa.set_ylim([15,20])
            plt.xlabel("Range [m]", fontsize = 16)
            plt.ylabel("Azimuth [m]", fontsize = 16)
            cbar.set_label("Intensity [dB]", fontsize = 14)
            axa.set_aspect('equal')
        
    plt.show()
    


if plot_polarimetry == True:
    for i in range(4):
        name_mat = 'HVRT_' + name + '_matrix' + pol[i] + '.npy'
        I = np.load('I_mat_' + name_mat)
        I_mat[:,:,i] = I
    I_HH = I_mat[:,:,0]
    I_HV = (I_mat[:,:,1])# + I_mat[:,:,2])/2
    I_VV = I_mat[:,:,3]
    
    if pol_dB == True:
        I_HH = I_mat_dB[:,:,0]
        I_HV = I_mat_dB[:,:,1] # + I_mat[:,:,2])/2
        I_VV = I_mat_dB[:,:,3]
        min_HH = np.min(I_HH)
        min_HV = np.min(I_HV)
        min_VV = np.min(I_VV)
        min_tot = np.min([min_HH, min_HV, min_VV])
        
        I_HH = I_HH - min_tot
        I_HV = I_HV - min_tot
        I_VV = I_VV - min_tot
        
    
    if multilook_var == True:
        I_HH = multilook(I_HH, Multi_dim)
        I_HV = multilook(I_HV, Multi_dim)
        I_VV = multilook(I_VV, Multi_dim)
        
        N_i = int(len(I_VV[0,:,]))
        x_vec = np.linspace(x_min,x_max,N_i)
        y_vec = np.linspace(y_min,y_max,N_i)
        
        
    if plot_SAR_final == True:
        fig = plt.figure(figsize=(12,6))
        gs = fig.add_gridspec(1, 3, hspace=0.2, wspace = 0.4)
        (ax1, ax2, ax3) = gs.subplots(sharex=True, sharey=True)
        ax1.set_title('HH', fontsize=16)
        ax2.set_title('HV', fontsize=16)
        ax3.set_title('VV', fontsize=16)
        plt1 = ax1.pcolormesh(x_vec, y_vec, np.absolute(I_HH[0:I_HH[:,1].size-1,0:I_HH[1,:].size-1])*100, cmap='gray')
        plt2 = ax2.pcolormesh(x_vec, y_vec, np.absolute(I_HV[0:I_HV[:,1].size-1,0:I_HV[1,:].size-1])*100, cmap='gray')
        plt3 = ax3.pcolormesh(x_vec, y_vec, np.absolute(I_VV[0:I_VV[:,1].size-1,0:I_VV[1,:].size-1])*100, cmap='gray')
        ax1.set_aspect('equal')
        ax2.set_aspect('equal')
        ax3.set_aspect('equal')
        #ax1.set_yticks([y_min,y_max])
        ax1.tick_params(labelsize=14)
        ax2.tick_params(labelsize=14)
        ax3.tick_params(labelsize=14)
        ax1.set_xlabel('Range [m]', fontsize=16)
        ax1.set_ylabel('Azimuth [m]', fontsize=16)
        ax2.set_xlabel('Range [m]', fontsize=16)
        ax3.set_xlabel('Range [m]', fontsize=16)
        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
        cbar = fig.colorbar(plt3, cax=cbar_ax)
        cbar.set_label("Intensity [a.u.]", fontsize=14)    
    
    
    max_value_vec = np.zeros(3)
    max_value_vec[0] = np.max(abs(I_HH))
    max_value_vec[1] = np.max(abs(I_HV))
    max_value_vec[2] = np.max(abs(I_VV))
    max_val = np.min(max_value_vec)
    rgb_scale = 1 / max_val


    blue = np.absolute(I_HH)*rgb_scale*1.2
    green = np.absolute(I_HV)*rgb_scale*1.2
    red = np.absolute(I_VV)*rgb_scale*1.2

    RED = np.zeros((int(len(blue[0,:])), int(len(blue[:,0])), 3))
    GREEN = np.zeros((int(len(blue[0,:])), int(len(blue[:,0])), 3))
    BLUE = np.zeros((int(len(blue[0,:])), int(len(blue[:,0])), 3))

    RED[:,:,0] = np.flip(red,0)
    GREEN[:,:,1] = np.flip(green,0)
    BLUE[:,:,2] = np.flip(blue,0)

    RGB = np.zeros((int(len(blue[0,:])), int(len(blue[:,0])), 3))
    RGB[:,:,0] = np.flip(red,0)
    RGB[:,:,1] = np.flip(green,0)
    RGB[:,:,2] = np.flip(blue,0)

    # plt.imshow(RGB)
    # plt.show()
    if plot_RGB == True:
        fig, axa = plt.subplots()
        axa.set_title('polSAR: Village ', fontsize=16)
        fpf=axa.imshow(RED, extent=[x_min,x_max,y_min,y_max])
        plt.xlabel("Range [m]", fontsize=16)
        plt.ylabel("Azimuth [m]", fontsize=16)
        axa.tick_params(labelsize=14)
        axa.set_aspect('equal')
        
        fig, axa = plt.subplots()
        axa.set_title('polSAR: Village ', fontsize=16)
        fpf=axa.imshow(GREEN, extent=[x_min,x_max,y_min,y_max])
        plt.xlabel("Range [m]", fontsize=16)
        plt.ylabel("Azimuth [m]", fontsize=16)
        axa.tick_params(labelsize=14)
        axa.set_aspect('equal')
        
        fig, axa = plt.subplots()
        axa.set_title('polSAR: Village ', fontsize=16)
        fpf=axa.imshow(BLUE, extent=[x_min,x_max,y_min,y_max])
        plt.xlabel("Range [m]", fontsize=16)
        plt.ylabel("Azimuth [m]", fontsize=16)
        axa.tick_params(labelsize=14)
        axa.set_aspect('equal')
        
    
    fig, axa = plt.subplots()
    axa.set_title('polSAR: Village ', fontsize=16)
    fpf=axa.imshow(RGB, extent=[x_min,x_max,y_min,y_max])
    plt.xlabel("Range [m]", fontsize=16)
    plt.ylabel("Azimuth [m]", fontsize=16)
    axa.tick_params(labelsize=14)
    axa.set_aspect('equal')
    plt.show()
    
    # fig = plt.figure(figsize=(12,6))
    # gs = fig.add_gridspec(1, 3, hspace=0.2, wspace = 0.4)
    # (ax1, ax2, ax3) = gs.subplots(sharex=True, sharey=True)
    # ax1.set_title('HH', fontsize=16)
    # ax2.set_title('HV', fontsize=16)
    # ax3.set_title('VV', fontsize=16)
    # plt1 = ax1.imshow(BLUE, extent=[x_min,x_max,y_min,y_max])
    # plt2 = ax2.imshow(GREEN, extent=[x_min,x_max,y_min,y_max])
    # plt3 = ax3.imshow(RED, extent=[x_min,x_max,y_min,y_max])
    # ax1.set_aspect('equal')
    # ax2.set_aspect('equal')
    # ax3.set_aspect('equal')
    # # ax1.set_yticks([-5,0,5])
    # # ax1.set_xticks([-5,0,5])
    # # ax2.set_xticks([-5,0,5])
    # # ax3.set_xticks([-5,0,5])

    # ax1.tick_params(labelsize=14)
    # ax2.tick_params(labelsize=14)
    # ax3.tick_params(labelsize=14)

    # ax1.set_xlabel('Range [m]', fontsize=16)
    # ax1.set_ylabel('Azimuth [m]', fontsize=16)
    # ax2.set_xlabel('Range [m]', fontsize=16)
    # ax3.set_xlabel('Range [m]', fontsize=16)
    
    