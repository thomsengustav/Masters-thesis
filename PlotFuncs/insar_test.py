# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:47:10 2025

@author: thoms
"""

'''
inSAR test
'''

from kamui import unwrap_dimensional
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def kernel_fkt(dim):
    kernel = np.ones((dim,dim)) / (dim*dim)
    return kernel

def phase_fkt(insar):
    Nx, Ny = int(len(insar[0,:])), int(len(insar[:,0]))
    phase = np.zeros((Nx,Ny))
    for i in range(Nx):
        for j in range(Ny):
            phase[i,j] = (insar[i,j] + np.pi) % (2 * np.pi) - np.pi
    return phase

phase1 = np.load('plancone_25_2deg.npy')
phase2 = np.load('plancone_25deg.npy')

from skimage.restoration import unwrap_phase


# phase1 = np.load('plan3cone45_2deg_fin_ops.npy')
# phase2 = np.load('plan3cone45deg_fin_ops.npy')

# phase1 = phase1[900:2400, 900:2400]
# phase2 = phase2[900:2400, 900:2400]

rad1 = 25.2/180*np.pi
rad2 = 25/180*np.pi

# rad1 = 45.2/180*np.pi
# rad2 = 45/180*np.pi
k = 201

plt.pcolor(np.angle(phase1), cmap='twilight_shifted')
plt.axis('equal')
plt.show()
plt.pcolor(np.angle(phase2), cmap='twilight_shifted')
plt.axis('equal')
plt.show()


fig, axa = plt.subplots()
axa.set_title('phase 25.2$\degree$')
fpf=axa.pcolormesh(np.angle(phase1), cmap='twilight_shifted')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("x [m]", fontsize=16)
plt.ylabel("y [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("phase [Rad]", fontsize=14)
axa.set_aspect('equal')

fig, axa = plt.subplots()
axa.set_title('phase 25$\degree$')
fpf=axa.pcolormesh(np.angle(phase2), cmap='twilight_shifted')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("x [m]", fontsize=16)
plt.ylabel("y [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("phase [Rad]", fontsize=14)
axa.set_aspect('equal')

plt.pcolor(abs(phase1), cmap='gray')
plt.show()

insar = phase1*np.conj(phase2)
plt.pcolor(np.angle(insar), cmap='twilight_shifted')
plt.title('raw phase')
plt.show()

fig, axa = plt.subplots()
axa.set_title('interferogram')
fpf=axa.pcolormesh(np.angle(insar), cmap='twilight_shifted')
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("x [m]", fontsize=16)
plt.ylabel("y [m]", fontsize=16)
axa.tick_params(labelsize=14)
cbar.set_label("phase [Rad]", fontsize=14)
axa.set_aspect('equal')
plt.show()


insar_pu = unwrap_dimensional(np.angle(insar))
plt.pcolor(insar_pu, cmap = 'gray')
plt.title('PU phase')
cbar = plt.colorbar()
plt.show()

smooth_phase = np.zeros((400,400), dtype = np.complex128)
smooth_phase = signal.convolve2d(insar, kernel_fkt(5), mode='same')
plt.pcolor(np.angle(smooth_phase), cmap='twilight_shifted')
plt.title('smoothed_ phase')
cbar = plt.colorbar()
plt.show()

insar_pu2 = unwrap_phase(np.angle(insar))
plt.pcolor(insar_pu2, cmap = 'gray')
plt.title('PU phase skimage')
cbar = plt.colorbar()
plt.show()

val = k * (np.cos(rad2)-np.cos(rad1))

height = insar_pu / val
plt.pcolor(height, cmap = 'gray')
cbar = plt.colorbar()
plt.show()



# different kernels for smoothing
kernel = 1/9 * np.array([[ 1, 1, 1],
                   [1, 1, 1],
                   [1, 1, 1]])


kernel_gaus_3x3 = 1/16 * np.array([[1, 2, 1],
                                  [2, 4, 1],
                                  [1, 2, 1]])

kernel_gaus_5x5 = 1/273 * np.array([[1, 4, 7, 4, 1],
                                   [4, 16, 26, 16, 4],
                                   [7, 26, 41, 26, 7],
                                   [4, 16, 26, 16, 4],
                                   [1, 4, 7, 4, 1]])

kernel_gaus_7x7 = 1/1003 * np.array([[0, 0, 1, 2, 1, 0, 0],
                                   [0, 3, 13, 22, 13, 3, 0],
                                   [1, 12, 59, 97, 59, 13, 1],
                                   [2, 22, 97, 159, 97, 22, 2],
                                   [1, 12, 59, 97, 59, 13, 1],
                                   [0, 3, 13, 22, 13, 3, 0],
                                   [0, 0, 1, 2, 1, 0, 0]])




def compute_coherence(image1, image2, window_size = 5):
    kernel = kernel_fkt(window_size)
    
    cross_corr = signal.convolve2d(image1 * np.conj(image2), kernel, boundary='symm', mode = 'same')
    
    power1 = signal.convolve2d(np.abs(image1)**2, kernel, boundary='symm', mode = 'same')
    power2 = signal.convolve2d(np.abs(image2)**2, kernel, boundary='symm', mode = 'same')
    
    coherence_map = np.abs(cross_corr) / (np.sqrt(power1 * power2) + 1e-6)
    return coherence_map

coherence_map = compute_coherence(phase1, phase2, window_size = 5)
plt.pcolor(coherence_map, cmap = 'gray')
plt.title('coherence_map')
plt.show()


cutoff_coherence = 0.1 # values below are removed from raw data


smooth2 = signal.convolve2d(height, kernel_fkt(5), boundary='symm', mode='same')

# smooth2 = signal.convolve2d(height, kernel_gaus_7x7, boundary='symm', mode='same')
def cut_w_coherence(cutoff_coherence, coherence_map, insarimage):
    new_image = np.zeros((int(len(insarimage[0,:])),int(len(insarimage[:,0]))))
    mask = np.zeros((int(len(insarimage[0,:])),int(len(insarimage[:,0]))))
    N = int(len(insarimage[0,:]))
    M = int(len(insarimage[1,:]))
    for j in range(M):
        for i in range(N):
            value = coherence_map[i,j]
            if value > cutoff_coherence:
                new_image[i,j] = insarimage[i,j]
                mask[i,j] = 1
    return new_image, mask

new_mat = smooth2[100:300,:200]

plt.pcolor(new_mat)
cbar = plt.colorbar()
plt.show()

x, y = np.meshgrid(range(new_mat.shape[0]), range(new_mat.shape[1]))
c = phase_fkt(new_mat * val)



# plt.pcolor(smooth, cmap = 'gray')
# cbar = plt.colorbar()
# plt.show()

# line = height[200,:]
# plt.plot(line)
# plt.show()

smooth_abs = signal.convolve2d(insar_pu, kernel, boundary='symm', mode='same')

plt.pcolor(smooth_abs, cmap = 'gray')
cbar = plt.colorbar()
plt.show()

# heiht_smooth = smooth_abs / val
# plt.pcolor(heiht_smooth, cmap = 'gray')
# cbar = plt.colorbar()
# plt.show()

# line2 = heiht_smooth[200,:]
# plt.plot(line2)
# plt.show()

phases = phase_fkt(smooth_abs)
plt.pcolor(phases, cmap = 'twilight_shifted')
plt.title('smoothed phase')
plt.show()


new_image, mask = cut_w_coherence(cutoff_coherence, coherence_map, phases)

plt.pcolor(new_image, cmap = 'twilight_shifted')
plt.title('phase_w_cutoff')
plt.show()


plt.pcolor(mask, cmap = 'gray')
plt.title('coherence mask')
plt.show()


insar_pu_new = unwrap_dimensional(new_image)

plt.pcolor(insar_pu_new, cmap = 'gray')
plt.title('PU abs phase w. mask')
cbar = plt.colorbar()
plt.show()

height_mask = insar_pu_new / val

plt.pcolor(height_mask, cmap = 'gray')
plt.title('height w. mask')
cbar = plt.colorbar()
plt.show()


smooth_cutoff = signal.convolve2d(height_mask, kernel_fkt(5), boundary='symm', mode='same')

plt.pcolor(smooth_cutoff, cmap = 'gray')
plt.title('smoothed height w. mask')
cbar = plt.colorbar()
plt.show()

import matplotlib.pyplot as plt
# This import registers the 3D projection, but is otherwise unused.

from plot_height_with_fringes import plot_height_with_fringes

plot_height_with_fringes(x, y, new_mat, c, 35, axis_equal=False)

x, y = np.meshgrid(range(insar_pu_new.shape[0]), range(insar_pu_new.shape[1]))
c = phase_fkt(smooth_cutoff * val)

plot_height_with_fringes(x, y, smooth_cutoff, c, 25, 135, axis_equal=False)
