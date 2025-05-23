# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 11:00:34 2024

@author: thoms
"""

#### plot vectors function
'''
Inputs: vector components as x, y, z vectors, vector_origin (1x3), elevation_angle, azimuth_angle, roll_angle, 
    xlim (1x2), ylim (1x2), zlim (1x2)

Output: plot of given vectors 
'''
import numpy as np
import matplotlib.pylab as plt

def plot_vector_fkt(x, y, z, vector_origin, xlim, ylim, zlim, elevation_angle, azimuth_angle, roll_angle):
    ax = plt.figure().add_subplot(projection='3d')

    for i in range(len(x)):
        ax.plot([vector_origin[0], x[i]] ,[vector_origin[1], y[i]], [vector_origin[2], z[i]])
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[0], zlim[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=elevation_angle, azim=azimuth_angle, roll=roll_angle)
    plt.show()
    
def D2_plot_vector_fkt(x, y, vector_origin, xlim, ylim):
    ax = plt.figure().add_subplot()

    for i in range(len(x)):
        ax.plot([vector_origin[0], x[i]] ,[vector_origin[1], y[i]])
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()

def plot_mesh_and_normal(Tri_mesh, normal_mat, xlim, ylim, zlim, elevation_angle, azimuth_angle, roll_angle):
    ax = plt.figure().add_subplot(projection='3d')

    for i in range(len(Tri_mesh[0,0,:])):
        ax.plot([Tri_mesh[0,0,i], Tri_mesh[1,0,i], Tri_mesh[2,0,i], Tri_mesh[0,0,i]],[Tri_mesh[0,1,i], Tri_mesh[1,1,i], Tri_mesh[2,1,i], Tri_mesh[0,1,i]], [Tri_mesh[0,2,i], Tri_mesh[1,2,i], Tri_mesh[2,2,i], Tri_mesh[0,2,i]],'b')
        # centroid = (Tri_mesh[0,:,i] + Tri_mesh[1,:,i] + Tri_mesh[2,:,i]) / 3
        # ax.plot([centroid[0], centroid[0] + normal_mat[i,0]], [centroid[1], centroid[1] + normal_mat[i,1]], [centroid[2], centroid[2] + normal_mat[i,2]])
        
    
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[0], zlim[1])
    ax.set_xlabel('X [m]', fontsize=16)
    ax.set_ylabel('Y [m]',fontsize=16)
    ax.set_zlabel('',fontsize=16)

    ax.view_init(elev=elevation_angle, azim=azimuth_angle, roll=roll_angle)
    plt.show()
    
def plot_tri_norm_rays_fkt(vecs, vec_origin, Tri_mesh, normal_mat, xlim, ylim, zlim, elevation_angle, azimuth_angle, roll_angle):
    ax = plt.figure().add_subplot(projection='3d')

    for i in range(len(vecs[:,0])):
        ax.plot([vec_origin[0], vec_origin[0] + vecs[i,0]] ,[vec_origin[1], vec_origin[1] + vecs[i,1]], [vec_origin[2], vec_origin[2]+ vecs[i,2]])
    
    for i in range(len(normal_mat[:,0])):
        ax.plot([Tri_mesh[0,0,i], Tri_mesh[1,0,i], Tri_mesh[2,0,i], Tri_mesh[0,0,i]],[Tri_mesh[0,1,i], Tri_mesh[1,1,i], Tri_mesh[2,1,i], Tri_mesh[0,1,i]], [Tri_mesh[0,2,i], Tri_mesh[1,2,i], Tri_mesh[2,2,i], Tri_mesh[0,2,i]])
        centroid = (Tri_mesh[0,:,i] + Tri_mesh[1,:,i] + Tri_mesh[2,:,i]) / 3
        ax.plot([centroid[0], centroid[0] + normal_mat[i,0]], [centroid[1], centroid[1] + normal_mat[i,1]], [centroid[2], centroid[2] + normal_mat[i,2]])
        
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_zlim(zlim[0], zlim[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    ax.view_init(elev=elevation_angle, azim=azimuth_angle, roll=roll_angle)
    plt.show()
    
    
    
    