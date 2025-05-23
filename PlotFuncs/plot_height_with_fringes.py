# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 14:31:45 2025

@author: thoms
"""

'''
function for plotting height maps with fringes
'''

import matplotlib.pyplot as plt

'''
all values need to be same size n x m matrices of the x, y, z, and phase values at each point

X_mesh, Y_mesh, Z_mesh, phase_mesh
'''

def plot_height_with_fringes(X_mesh, Y_mesh, Z_mesh, phase_mesh, elev = 30, azim = 135, axis_equal = True):
    fig, ax = plt.subplots(subplot_kw={'projection': '3d'})
    
    scamap = plt.cm.ScalarMappable(cmap='twilight_shifted')
    fcolors = scamap.to_rgba(phase_mesh)
    ax.plot_surface(X_mesh, Y_mesh, Z_mesh, facecolors=fcolors, cmap='twilight_shifted')
    
    # here you can change view direction with elev and azim.
    ax.view_init(elev=elev, azim=azim, roll=0)
    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')
    ax.set_zlim([-10,35])
    if axis_equal == True:
        ax.set_aspect('equal')
    #cbar = plt.colorbar(plt1, shrink=1, pad=0.15)
    cbar = fig.colorbar(scamap, pad=0.15)
    cbar.set_label("phase (rad)",fontsize=14)
    plt.show()