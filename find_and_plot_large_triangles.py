# -*- coding: utf-8 -*-
"""
Created on Thu May 22 10:07:04 2025

@author: thoms
"""

'''
code for finding and plotting large triangles in meshes
'''

import numpy as np
from fast_vec_operators import cross_fkt
import matplotlib.pyplot as plt


def find_large_tri(dist_mat, cen_tri_node_mat, NTri_in_leafs_vec, max_dist):
    tri_tot = int(np.sum(NTri_in_leafs_vec))
    leafs = int(len(cen_tri_node_mat[0,0,0,:]))
    v1_large = np.where(dist_mat[0,:] > max_dist)[0]
    v2_large = np.where(dist_mat[1,:] > max_dist)[0]
    v3_large = np.where(dist_mat[2,:] > max_dist)[0]
    large_tri_N = int(len(v1_large) + len(v2_large) + len(v3_large))
    large_tri_mat = np.zeros((4,3,large_tri_N))
    tri_mat = np.zeros((4,3,tri_tot))
    counter = 0
    tri_counter = 0
    for i in range(leafs):
        for n in range(int(NTri_in_leafs_vec[i])):
            tri_mat[:,:,counter] = cen_tri_node_mat[:,:,n,i]
            counter += 1
    
    for i in range(int(len(v1_large))):
        large_tri_mat[:,:,tri_counter] = tri_mat[:,:,v1_large[i]]
        tri_counter += 1
        
    for i in range(int(len(v2_large))):
        large_tri_mat[:,:,tri_counter] = tri_mat[:,:,v2_large[i]]
        tri_counter += 1
        
    for i in range(int(len(v3_large))):
        large_tri_mat[:,:,tri_counter] = tri_mat[:,:,v3_large[i]]
        tri_counter += 1

    return large_tri_mat

# large_tri_mat = find_large_tri(dist_mat, cen_tri_node_mat, NTri_in_leafs_vec, 1)


def plot_large_tri(large_tri_mat, xlim_v, ylim_v, zlim_v, elevation_angle, azimuth_angle):
    counter = 0
    for i in range(int(len(large_tri_mat[0,0,:]))):
        V1 = large_tri_mat[0,:,i]
        V2 = large_tri_mat[1,:,i]
        V3 = large_tri_mat[2,:,i]
        
        norm = cross_fkt(V1-V3,V2-V3)
       
        bad_vertex_mat = np.zeros((3,3,int(len(large_tri_mat[0,0,:]))))
        if norm[0] == 0 and norm[1] == 0 and norm[2] == 0:
            bad_vertex_mat[0,:,counter] = V1
            bad_vertex_mat[1,:,counter] = V2
            bad_vertex_mat[2,:,counter] = V3
            # print('triangle')
            # print(V1)
            # print(V2)
            # print(V3)
            counter += 1
            
            ax = plt.figure().add_subplot(projection='3d')
            
            x = [large_tri_mat[0,0,i], large_tri_mat[1,0,i], large_tri_mat[2, 0,i], large_tri_mat[0,0,i]]
            y = [large_tri_mat[0,1,i], large_tri_mat[1,1,i], large_tri_mat[2, 1,i], large_tri_mat[0,1,i]]
            z = [large_tri_mat[0,2,i], large_tri_mat[1,2,i], large_tri_mat[2, 2,i], large_tri_mat[0,2,i]]
            ax.plot(x ,y, z)
            #ax.scatter(large_tri_mat[3,0,i], large_tri_mat[3,1,i], large_tri_mat[3,2,i])
            
            min_x = np.min(x)
            max_x = np.max(x)
            min_y = np.min(y)
            max_y = np.max(y)
            min_z = np.min(z)
            max_z = np.max(z)
            delta = 0.05
            
            ax.set_xlim([min_x-delta, max_x+delta])
            ax.set_ylim([min_y-delta, max_y+delta])
            ax.set_zlim([min_z-delta, max_z+delta])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
    
            ax.view_init(elev=elevation_angle, azim=azimuth_angle, roll=0)
            plt.show()
    return bad_vertex_mat, counter
# mat_var, counter_Var = plot_large_tri(Triangle_matrix, np.array([-5,5]), np.array([-5,5]), np.array([0,10]), 30, 30)
