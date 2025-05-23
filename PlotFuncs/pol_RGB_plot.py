# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 14:00:00 2025

@author: thoms
"""


import numpy as np
import matplotlib.pyplot as plt
path = "C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\"

I_HH = np.load(path + 'I_mat_all_box_2_HH.npy')
I_HV = np.load(path + 'I_mat_all_box_2_HV.npy')
I_VV = np.load(path + 'I_mat_all_box_2_VV.npy')
I_VH = np.load(path + 'I_mat_all_box_2_VH.npy')

I_HH = np.load(path + 'I_mat_all_råsted_1_HH.npy')
I_HV = np.load(path + 'I_mat_all_råsted_1_HV.npy')
I_VV = np.load(path + 'I_mat_all_råsted_1_VV.npy')
I_VH = np.load(path + 'I_mat_all_råsted_1_VH.npy')

x_vec=np.linspace(-5, 5, num=200) #Scene center er sat i (x=0,y=0)
y_vec=np.linspace(-5, 5, num=200) #x og y interval i meter, num=antal pixels i x og y dim

fig, axa = plt.subplots()
axa.set_title('polSAR: HH')
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I_HH[0:I_HH[:,1].size-1,0:I_HH[1,:].size-1]), cmap='gray')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')

fig, axa = plt.subplots()
axa.set_title('polSAR: HV')
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I_HV[0:I_HV[:,1].size-1,0:I_HV[1,:].size-1]), cmap='gray')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')

fig, axa = plt.subplots()
axa.set_title('polSAR: VV')
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I_VV[0:I_VV[:,1].size-1,0:I_VV[1,:].size-1]), cmap='gray')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')
plt.show()

fig, axa = plt.subplots()
axa.set_title('VH')
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I_VH[0:I_VH[:,1].size-1,0:I_VH[1,:].size-1]), cmap='gray')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("Intensity [a.u.]", fontsize=14)
axa.set_aspect('equal')
plt.show()

fig = plt.figure(figsize=(12,6))
gs = fig.add_gridspec(1, 3, hspace=0.2, wspace = 0.4)
(ax1, ax2, ax3) = gs.subplots(sharex=True, sharey=True)
# fig.suptitle('Difference between \n Ray Tracing and Rasterization simulations', fontsize=16)
ax1.set_title('HH', fontsize=16)
ax2.set_title('HV', fontsize=16)
ax3.set_title('VV', fontsize=16)
#ax3.set_title('Difference')
plt1 = ax1.pcolormesh(y_vec, x_vec, np.absolute(I_HH[0:I_HH[:,1].size-1,0:I_HH[1,:].size-1])*1000, cmap='gray')
plt2 = ax2.pcolormesh(y_vec, x_vec, np.absolute(I_HV[0:I_HV[:,1].size-1,0:I_HV[1,:].size-1])*1000, cmap='gray')
plt3 = ax3.pcolormesh(y_vec, x_vec, np.absolute(I_VV[0:I_VV[:,1].size-1,0:I_VV[1,:].size-1])*1000, cmap='gray')
#ax3.pcolormesh(x_vec, y_vec, I_diff_scale)
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax1.set_yticks([-5,0,5])
# ax1.set_yticks([-1,0,1])
# ax2.set_xticks([0,1,2])
# ax2.set_yticks([-1,0,1])
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
ax3.tick_params(labelsize=14)
#ax3.set_aspect('equal')
ax1.set_xlabel('Range [m]', fontsize=16)
ax1.set_ylabel('Azimuth [m]', fontsize=16)
ax2.set_xlabel('Range [m]', fontsize=16)
ax3.set_xlabel('Range [m]', fontsize=16)
# cbar2 = plt.colorbar(plt3,ax=ax3, shrink=0.6)
# cbar2.set_label("Intensity [a.u.]", fontsize=14)
fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
cbar = fig.colorbar(plt3, cax=cbar_ax)
cbar.set_label("Intensity [a.u.]", fontsize=14)

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])






I_HV_VH = (I_HV + I_VH)/2

fig, axa = plt.subplots()
axa.set_title('VH_HV_div2')
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I_HV_VH[0:I_HV_VH[:,1].size-1,0:I_HV_VH[1,:].size-1]), cmap='gray')
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
max_value_vec[0] = np.max(np.abs(I_HH))
max_value_vec[1] = np.max(np.abs(I_HV))
max_value_vec[2] = np.max(np.abs(I_VV))
max_val = np.max(max_value_vec)

rgb_scale = 1 / max_val


blue = np.absolute(I_HH)*rgb_scale*1.6
green = np.absolute(I_HV)*rgb_scale*1.6
red = np.absolute(I_VV)*rgb_scale*1.6


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

ticks = [-5,0,5]

fig = plt.figure(figsize=(12,6))
gs = fig.add_gridspec(1, 3, hspace=0.2, wspace = 0.4)
(ax1, ax2, ax3) = gs.subplots(sharex=True, sharey=True)
# fig.suptitle('Difference between \n Ray Tracing and Rasterization simulations', fontsize=16)
ax1.set_title('HH', fontsize=16)
ax2.set_title('HV', fontsize=16)
ax3.set_title('VV', fontsize=16)
#ax3.set_title('Difference')
plt1 = ax1.imshow(BLUE, extent=[-5,5,-5,5])
plt2 = ax2.imshow(GREEN, extent=[-5,5,-5,5])
plt3 = ax3.imshow(RED, extent=[-5,5,-5,5])
#ax3.pcolormesh(x_vec, y_vec, I_diff_scale)
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax1.set_yticks([-5,0,5])
ax1.set_xticks([-5,0,5])
ax2.set_xticks([-5,0,5])
ax3.set_xticks([-5,0,5])

ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
ax3.tick_params(labelsize=14)

ax1.set_xlabel('Range [m]', fontsize=16)
ax1.set_ylabel('Azimuth [m]', fontsize=16)
ax2.set_xlabel('Range [m]', fontsize=16)
ax3.set_xlabel('Range [m]', fontsize=16)
# cbar2 = plt.colorbar(plt3,ax=ax3, shrink=0.6)
# cbar2.set_label("Intensity [a.u.]", fontsize=14)
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.3, 0.02, 0.4])
# cbar = fig.colorbar(plt3, cax=cbar_ax)
# cbar.set_label("Intensity [a.u.]", fontsize=14)


fig, axa = plt.subplots()
axa.set_title('polSAR: village ', fontsize=16)
fpf=axa.imshow(RGB, extent=[-5,5,-5,5])
plt.xlabel("Range [m]", fontsize=16)
plt.ylabel("Azimuth [m]", fontsize=16)
axa.tick_params(labelsize=14)
axa.set_aspect('equal')
plt.show()
               


