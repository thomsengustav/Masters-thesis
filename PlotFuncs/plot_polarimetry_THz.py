# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 09:28:35 2025

@author: thoms
"""

'''
plot polarimetry data from setup
'''

import numpy as np
import matplotlib.pyplot as plt


plot_RGB = True

name = 'POLSAR_bil'
ending = '.npy'

S_HH = np.load('HH_' + name + ending)
S_HV = np.load('HV_' + name + ending)
S_VH = np.load('VH_' + name + ending)
S_VV = np.load('VV_' + name + ending)


S_HH = S_HH[200:600,100:700]
S_HV = S_HV[200:600,100:700]
S_VH = S_VH[200:600,100:700]
S_VV = S_VV[200:600,100:700]


import matplotlib.colors 
custom_map_linlog = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#000000", "#3F3F3F","#737373","#989898","#B3B3B3","#C8C8C8","#D8D8D8","#E6E6E6","#F3F3F3","#FFFFFF"])

Nx = int(len(S_HH[0,:]))
Ny = int(len(S_HH[:,0]))

x_min = -3
x_max = 3
y_min = -2
y_max = 2

x_vec = np.linspace(x_min, x_max, Nx)
y_vec = np.linspace(y_min, y_max, Ny)

fig, axa = plt.subplots()
axa.set_title('Clutter_HH')
fpf=axa.pcolormesh(x_vec, y_vec, np.absolute(S_HH[0:S_HH[:,1].size-1,0:S_HH[1,:].size-1])*100, cmap=custom_map_linlog)
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')

fig, axa = plt.subplots()
axa.set_title('Clutter_HV')
fpf=axa.pcolormesh(x_vec, y_vec, np.absolute(S_HV[0:S_HV[:,1].size-1,0:S_HV[1,:].size-1])*100, cmap=custom_map_linlog)
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')

fig, axa = plt.subplots()
axa.set_title('Clutter_VH')
fpf=axa.pcolormesh(x_vec, y_vec, np.absolute(S_VH[0:S_VH[:,1].size-1,0:S_VH[1,:].size-1])*100, cmap=custom_map_linlog)
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')

fig, axa = plt.subplots()
axa.set_title('Clutter_VV')
fpf=axa.pcolormesh(x_vec, y_vec, np.absolute(S_VV[0:S_VV[:,1].size-1,0:S_VV[1,:].size-1])*100, cmap=custom_map_linlog)
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')
plt.show()



fig = plt.figure(figsize=(9.5,8))
gs = fig.add_gridspec(2, 2, hspace=-0.2, wspace = 0.1)
((ax1, ax2), (ax3, ax4)) = gs.subplots(sharex=True, sharey=True)
ax1.set_title('HH', fontsize=16)
ax2.set_title('HV', fontsize=16)
ax3.set_title('VH', fontsize=16)
ax4.set_title('VV', fontsize=16)
plt1 = ax1.pcolormesh(x_vec, y_vec, np.absolute(S_HH[0:S_HH[:,1].size-1,0:S_HH[1,:].size-1])*100, cmap=custom_map_linlog)
plt2 = ax2.pcolormesh(x_vec, y_vec, np.absolute(S_HV[0:S_HV[:,1].size-1,0:S_HV[1,:].size-1])*100, cmap=custom_map_linlog)
plt3 = ax3.pcolormesh(x_vec, y_vec, np.absolute(S_VH[0:S_VH[:,1].size-1,0:S_VH[1,:].size-1])*100, cmap=custom_map_linlog)
plt4 = ax4.pcolormesh(x_vec, y_vec, np.absolute(S_VV[0:S_VV[:,1].size-1,0:S_VV[1,:].size-1])*100, cmap=custom_map_linlog)
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')
#ax1.set_yticks([y_min,y_max])
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
ax3.tick_params(labelsize=14)
ax4.tick_params(labelsize=14)
ax1.set_ylabel('Azimuth [cm]', fontsize=16)
ax3.set_xlabel('Range [cm]', fontsize=16)
ax3.set_ylabel('Azimuth [cm]', fontsize=16)
ax4.set_xlabel('Range [cm]', fontsize=16)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.83, 0.205, 0.03, 0.56])
cbar = fig.colorbar(plt3, cax=cbar_ax)
cbar.set_label("Intensity [a.u.]", fontsize=16)    
plt.show()

def get_S_mat():
    Nx = int(len(S_HH[0,:]))
    Ny = int(len(S_HH[:,0]))

    S_mat = np.zeros((Ny,Nx,4), dtype = np.complex128)
    S_mat[:,:,0] = S_HH
    S_mat[:,:,1] = S_HV
    S_mat[:,:,2] = S_VH
    S_mat[:,:,3] = S_VV
    return S_mat

S_mat = get_S_mat()


I_HH = S_mat[:,:,0]
I_HV = S_mat[:,:,1]
I_VV = S_mat[:,:,3]

    
def zoom(SAR_image, zoom, x_vec, y_vec, x_off, y_off):
    N = int(len(SAR_image[0,:]))
    N_n = int(np.floor(N / zoom))
    SAR_n = np.zeros((N_n, N_n), dtype = np.complex128)
    
    x_max_n = np.max(x_vec) / zoom
    x_min_n = np.min(x_vec) / zoom
    y_max_n = np.max(y_vec) / zoom
    y_min_n = np.min(y_vec) / zoom
    
    y_n = np.linspace(y_min_n, y_max_n, N_n)
    x_n = np.linspace(x_min_n, x_max_n, N_n)
    
    for i in range(N_n):
        for n in range(N_n):
            SAR_n[i,n] =  SAR_image[i+x_off,n+y_off]
    
    return SAR_n, x_n, y_n    
    

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

# I_HH = multilook(I_HH, 5)
# I_HV = multilook(I_HV, 5)
# I_VV = multilook(I_VV, 5)

x_off = 166
y_off = 166

x_max = np.max(x_vec)
y_max = np.max(y_vec)
x_min = np.min(x_vec)
y_min = np.min(y_vec)

max_value_vec = np.zeros(3)
max_value_vec[0] = np.max(abs(I_HH))
max_value_vec[1] = np.max(abs(I_HV))
max_value_vec[2] = np.max(abs(I_VV))
max_val = np.min(max_value_vec)
rgb_scale = 1 / max_val


blue = np.absolute(I_HH)*rgb_scale*1.8
green = np.absolute(I_HV)*rgb_scale*1.8
red = np.absolute(I_VV)*rgb_scale*1.8

RED = np.zeros((Ny,Nx, 3))
GREEN = np.zeros((Ny,Nx, 3))
BLUE = np.zeros((Ny,Nx, 3))

RED[:,:,0] = red
GREEN[:,:,1] = green
BLUE[:,:,2] = blue


RGB = np.zeros((Ny, Nx, 3))
RGB[:,:,0] = red
RGB[:,:,1] = green
RGB[:,:,2] = blue


plt.imshow(RGB)
plt.show()
if plot_RGB == True:
    plt.imshow(RED)
    plt.title('VV')
    plt.show()
    plt.imshow(GREEN)
    plt.title('HV')
    plt.show()
    plt.imshow(BLUE)
    plt.title('HH')
    plt.show()


fig, axa = plt.subplots()
axa.set_title('polSAR: Car on clutter', fontsize=16)
fpf=axa.imshow(np.flip(RGB,0) , extent=[x_min,x_max,y_min,y_max])
plt.xlabel("Range [cm]", fontsize=16)
plt.ylabel("Azimuth [cm]", fontsize=16)
axa.tick_params(labelsize=14)
axa.set_aspect('equal')
plt.show()

