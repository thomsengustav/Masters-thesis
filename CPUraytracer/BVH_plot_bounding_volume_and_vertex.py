# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 09:50:58 2025

@author: thoms
"""

'''
Plot boundin volumes and vertex function
'''

import matplotlib.pylab as plt
from numba import njit
import numpy as np
from BVH_library_funcs import isin_fkt

@njit
def find_N_tri_in_box(box_index, new_index_array, NTri_in_leafs_vec):
    array_len = int(len(new_index_array[0,:]))
    stack = np.ones(array_len+1)*(999999)
    visited = np.ones(array_len)*(-1)
    if new_index_array[0,box_index] == -9:
        return 1
    stack[0] = new_index_array[0,box_index]
    stack[1] = new_index_array[1,box_index]
    current_node = 0
    tot_tri = 0
    while current_node != 999999:
        current_node = int(np.min(stack))
        if current_node == 999999:
            return tot_tri
        stack_index = np.where(stack == current_node)[0][0]
        #print(current_node)
        visited[current_node] = current_node
        stack[stack_index] = 999999
        child1_index = (new_index_array[0,current_node])
        child1_index = int(child1_index)
        child2_index = (new_index_array[1,current_node])
        child2_index = int(child2_index)
        if child1_index == -9:
            tot_tri += NTri_in_leafs_vec[child2_index]
        else:
            if isin_fkt(child1_index, visited) == False:
                stack[child1_index] = child1_index
            if isin_fkt(child2_index, visited) == False:
                stack[child2_index] = child2_index


@njit
def find_tri_in_box(box_index, new_index_array, NTri_in_leafs_vec, cen_tri_node_mat):
    array_len = int(len(new_index_array[0,:]))
    stack = np.ones(array_len+1)*(999999)
    visited = np.ones(array_len)*(-1)
    stack[0] = new_index_array[0,box_index]
    stack[1] = new_index_array[1,box_index]
    tri_in_box = find_N_tri_in_box(box_index, new_index_array, NTri_in_leafs_vec)
    new_cen_tri_node_mat = np.zeros((4,3,int(tri_in_box)))
    current_node = 0
    tri_counter = 0
    while current_node != 999999:
        current_node = int(np.min(stack))
        if current_node == 999999:
            return new_cen_tri_node_mat
        stack_index = np.where(stack == current_node)[0][0]
        visited[current_node] = current_node
        stack[stack_index] = 999999
        child1_index = (new_index_array[0,current_node])
        child1_index = int(child1_index)
        child2_index = (new_index_array[1,current_node])
        child2_index = int(child2_index)
        if child1_index == -9:
            N_tri = int(NTri_in_leafs_vec[child2_index])
            t_tri = np.zeros(N_tri)
            for i in range(N_tri):
                new_cen_tri_node_mat[0,:,tri_counter] = cen_tri_node_mat[0,:,i,child2_index]
                new_cen_tri_node_mat[1,:,tri_counter] = cen_tri_node_mat[1,:,i,child2_index]
                new_cen_tri_node_mat[2,:,tri_counter] = cen_tri_node_mat[2,:,i,child2_index]
                new_cen_tri_node_mat[3,:,tri_counter] = cen_tri_node_mat[3,:,i,child2_index]
                tri_counter += 1
        else:
            if isin_fkt(child1_index, visited) == False:
                stack[child1_index] = child1_index
            if isin_fkt(child2_index, visited) == False:
                stack[child2_index] = child2_index


def plot_BV_and_vertex(box_node_mat, box_index, new_index_array, NTri_in_leafs_vec, cen_tri_node_mat, xlim_v, ylim_v, zlim_v, elevation_angle, azimuth_angle):
    box = box_node_mat[:,box_index]
    ax = plt.figure().add_subplot(projection='3d')
    x_min = box[0]
    x_max = box[1]
    y_min = box[2]
    y_max = box[3]
    z_min = box[4]
    z_max = box[5]
    
    x_vec = [x_min, x_max, x_max, x_min, x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_max, x_max, x_min, x_min]
    y_vec = [y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_max, y_max]
    z_vec = [z_min, z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max, z_max, z_max, z_min, z_min, z_max, z_max, z_min]
    
    
    x_vec1 = [x_max, x_max, x_max, x_max, x_max, x_min, x_min, x_max]
    y_vec1 = [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max]
    z_vec1 = [z_max, z_min, z_min, z_max, z_max, z_max, z_max, z_max]
    
    x_vec2 = [x_max, x_min, x_min, x_min, x_min, x_max, x_min, x_min]
    y_vec2 = [y_min, y_min, y_min, y_min, y_max, y_max, y_max, y_max]
    z_vec2 = [z_min, z_min, z_max, z_min ,z_min, z_min, z_min, z_max]
    
    ax.plot(x_vec1 ,y_vec1, z_vec1, c='k', zorder=10)
    
    cen_in_box = find_tri_in_box(box_index, new_index_array, NTri_in_leafs_vec, cen_tri_node_mat)
    V1x, V1y, V1z = cen_in_box[0,0,:], cen_in_box[0,1,:], cen_in_box[0,2,:]
    V2x, V2y, V2z = cen_in_box[1,0,:], cen_in_box[1,1,:], cen_in_box[1,2,:]
    V3x, V3y, V3z = cen_in_box[2,0,:], cen_in_box[2,1,:], cen_in_box[2,2,:]
    V4x, V4y, V4z = cen_in_box[3,0,:], cen_in_box[3,1,:], cen_in_box[3,2,:]
    
    # ncenx = (V1x + V2x + V3x) / 3
    # nceny = (V1y + V2y + V3y) / 3 
    # ncenz = (V1z + V2z + V3z) / 3 
    ax.scatter(V1x, V1y, V1z, c='r', zorder=0)
    ax.scatter(V2x, V2y, V2z, c='r', zorder=0)
    ax.scatter(V3x, V3y, V3z, c='r', zorder=0)
    # ax.scatter(V4x, V4y, V4z, c='r', zorder=5, s=7)
    #ax.scatter(ncenx, nceny, ncenz, c='m', zorder=5)
    ax.plot(x_vec2 ,y_vec2, z_vec2, c='k', zorder=1)
    # ax.set_title('                                                                                 ')
    ax.set_xlim([xlim_v[0], xlim_v[1]])
    ax.set_ylim([ylim_v[0], ylim_v[1]])
    ax.set_zlim([zlim_v[0], zlim_v[1]])
    ax.set_xlabel('X [m]', fontsize = 14)
    ax.set_ylabel('Y [m]', fontsize = 14)
    ax.set_zlabel('Z [m]', fontsize = 14)
    ax.set_aspect('equal')
    plt.rcParams['axes.titley'] = 1.0
    plt.rcParams['axes.titlepad'] = -34
    #plt.colorbar(V1x)
    #ax.dist = 13
    ax.view_init(elev=elevation_angle, azim=azimuth_angle, roll=0)
    #ax.subplots_adjust(left=0.1, right=10, top=0.9, bottom=0.1)
    plt.show()
        
# plot_BV_and_vertex(box_node_mat, 0, new_index_array, NTri_in_leafs_vec, cen_tri_node_mat, np.array([-7,7]), np.array([-7,7]), np.array([0,10]), 90, 0)


# for i in range(128-1):
#     plot_BV_and_vertex(box_node_mat, i, new_index_array, NTri_in_leafs_vec, cen_tri_node_mat, np.array([-5,5]), np.array([-5,5]), np.array([0,10]), 20, -15)

def plot_BV_and_vertex_together(box_node_mat, box_index, new_index_array, NTri_in_leafs_vec, cen_tri_node_mat, xlim_v, ylim_v, zlim_v, elevation_angle, azimuth_angle, N):
    ax = plt.figure().add_subplot(projection='3d')
    box_indexloop = box_index
    for i in range(N):
        box = box_node_mat[:,box_indexloop]
        
        x_min = box[0]
        x_max = box[1]
        y_min = box[2]
        y_max = box[3]
        z_min = box[4]
        z_max = box[5]
        
        x_vec = [x_min, x_max, x_max, x_min, x_min, x_min, x_max, x_max, x_min, x_min, x_max, x_max, x_max, x_max, x_min, x_min]
        y_vec = [y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_min, y_min, y_min, y_max, y_max, y_max, y_max]
        z_vec = [z_min, z_min, z_min, z_min, z_min, z_max, z_max, z_max, z_max, z_max, z_max, z_min, z_min, z_max, z_max, z_min]
        
        
        x_vec1 = [x_max, x_max, x_max, x_max, x_max, x_min, x_min, x_max]
        y_vec1 = [y_min, y_min, y_max, y_max, y_min, y_min, y_max, y_max]
        z_vec1 = [z_max, z_min, z_min, z_max, z_max, z_max, z_max, z_max]
        
        x_vec2 = [x_max, x_min, x_min, x_min, x_min, x_max, x_min, x_min]
        y_vec2 = [y_min, y_min, y_min, y_min, y_max, y_max, y_max, y_max]
        z_vec2 = [z_min, z_min, z_max, z_min ,z_min, z_min, z_min, z_max]
        
        ax.plot(x_vec ,y_vec, z_vec, zorder=10)
        
        cen_in_box = find_tri_in_box(box_indexloop, new_index_array, NTri_in_leafs_vec, cen_tri_node_mat)
        V1x, V1y, V1z = cen_in_box[0,0,:], cen_in_box[0,1,:], cen_in_box[0,2,:]
        V2x, V2y, V2z = cen_in_box[1,0,:], cen_in_box[1,1,:], cen_in_box[1,2,:]
        V3x, V3y, V3z = cen_in_box[2,0,:], cen_in_box[2,1,:], cen_in_box[2,2,:]
        V4x, V4y, V4z = cen_in_box[3,0,:], cen_in_box[3,1,:], cen_in_box[3,2,:]
        
        # ax.scatter(V1x, V1y, V1z, c='r', zorder=0)
        # ax.scatter(V2x, V2y, V2z, c='r', zorder=0)
        # ax.scatter(V3x, V3y, V3z, c='r', zorder=0)
        # ax.scatter(V4x, V4y, V4z, c='r', zorder=5, s=7)
        #ax.scatter(ncenx, nceny, ncenz, c='m', zorder=5)
        #ax.plot(x_vec2 ,y_vec2, z_vec2, c='k', zorder=1)
        box_indexloop += 1
    # ax.set_title('                                                                                 ')
    ax.set_xlim([xlim_v[0], xlim_v[1]])
    ax.set_ylim([ylim_v[0], ylim_v[1]])
    ax.set_zlim([zlim_v[0], zlim_v[1]])
    # ax.set_xlabel('X [m]', fontsize = 14)
    # ax.set_ylabel('Y [m]', fontsize = 14)
    # ax.set_zlabel('Z [m]', fontsize = 14)
    ax.set_aspect('equal')
    plt.rcParams['axes.titley'] = 1.0
    plt.rcParams['axes.titlepad'] = -34
    #plt.colorbar(V1x)
    #ax.dist = 13
    ax.view_init(elev=elevation_angle, azim=azimuth_angle, roll=0)
    #ax.subplots_adjust(left=0.1, right=10, top=0.9, bottom=0.1)
    plt.axis('off')
    plt.show()