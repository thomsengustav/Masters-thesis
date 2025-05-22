# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:50:58 2025

@author: thoms
"""

'''
Plot boundin volumes function
'''

import matplotlib.pylab as plt

def plot_bounding_volume(index_array, box_node_mat, cen_tri_node_mat, N_diff_vec, xlim_v, ylim_v, zlim_v, elevation_angle, azimuth_angle, roll_angle):
    len_vec = len(N_diff_vec)
    N_accum=0
    for i in range(len_vec):
        ax = plt.figure().add_subplot(projection='3d')

        for n in range(int(N_diff_vec[i])):
            x_min = box_node_mat[0,n+N_accum]
            x_max = box_node_mat[1,n+N_accum]
            y_min = box_node_mat[2,n+N_accum]
            y_max = box_node_mat[3,n+N_accum]
            z_min = box_node_mat[4,n+N_accum]
            z_max = box_node_mat[5,n+N_accum]
            
            x_vec = [x_min, x_max, x_max, x_min, x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_max, x_max, x_min, x_min]
            y_vec = [y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_max, y_max]
            z_vec = [z_min, z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max, z_max, z_max, z_min, z_min, z_max, z_max, z_min]
            ax.plot(x_vec ,y_vec, z_vec)
            
        ax.set_xlim([xlim_v[0], xlim_v[1]])
        ax.set_ylim([ylim_v[0], ylim_v[1]])
        ax.set_zlim([zlim_v[0], zlim_v[1]])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        ax.view_init(elev=elevation_angle, azim=azimuth_angle, roll=roll_angle)
        plt.show()
        N_accum += int(N_diff_vec[i])