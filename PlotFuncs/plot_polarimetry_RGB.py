# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 12:01:27 2025

@author: thoms
"""

'''
Make polarimetry image test
'''

import numpy as np
import matplotlib.pyplot as plt

name = 'all_råsted_1'

I_HH = np.load('I_mat_' + name + '_HH.npy')
I_HV = np.load('I_mat_' + name + '_HV.npy')
I_VV = np.load('I_mat_' + name + '_VV.npy')
I_VH = np.load('I_mat_' + name + '_VH.npy')

x_vec=np.linspace(-4, 4, num=200) #Scene center er sat i (x=0,y=0)
y_vec=np.linspace(-5, 5, num=200) #x og y interval i meter, num=antal pixels i x og y dim

fig, axa = plt.subplots()
axa.set_title('HH')
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I_HH[0:I_HH[:,1].size-1,0:I_HH[1,:].size-1]), cmap='gray')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')

fig, axa = plt.subplots()
axa.set_title('HV')
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I_HV[0:I_HV[:,1].size-1,0:I_HV[1,:].size-1]), cmap='gray')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')

fig, axa = plt.subplots()
axa.set_title('VV')
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I_VV[0:I_VV[:,1].size-1,0:I_VV[1,:].size-1]), cmap='gray')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')
plt.show()

'''
HH -> blue, HV - > green, VV -> red
'''
max_value_vec = np.zeros(3)
max_value_vec = np.max(I_HH)
max_value_vec = np.max(I_HV)
max_value_vec = np.max(I_VV)
max_val = np.max(max_value_vec)

rgb_scale = 1 / (max_val)


blue = np.absolute(I_HH)*rgb_scale
green = np.absolute(I_HV)*rgb_scale
red = np.absolute(I_VV)*rgb_scale

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

plt.imshow(RGB)
plt.show()

plt.imshow(RED)
plt.show()
plt.imshow(GREEN)
plt.show()
plt.imshow(BLUE)
plt.show()



               


