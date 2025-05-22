# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 13:41:38 2024

@author: thoms
"""

'''
load .txt files and convert to trianglematrix np.array
'''


import numpy as np

### load RT ready mesh.txt

mesh_txt = np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\tonk72_RT_ready.txt")

# define triangle matrix
def tri_mat_fkt(mesh_txt):
    N_tris = int(len(mesh_txt[:,0])/3)
    Triangle_matrix = np.zeros((3,3,N_tris))
    
    for i in range(N_tris):
        Triangle_matrix[0,:,i] = mesh_txt[i*3,:]
        Triangle_matrix[1,:,i] = mesh_txt[i*3+1,:]
        Triangle_matrix[2,:,i] = mesh_txt[i*3+2,:]
    return Triangle_matrix


def calculate_normals(tri_mat):
    N_triangle = int(len(tri_mat[:,0,0]))
    triangle_normal_matrix = np.zeros((N_triangle,3))
    for i in range(N_triangle):
        V1 = tri_mat[0,:,i]
        V2 = tri_mat[1,:,i]
        V3 = tri_mat[2,:,i]

        norm = np.cross(V1-V3,V2-V3)
        norm = norm / np.linalg.norm(norm)
        triangle_normal_matrix[i,:] = norm
    return triangle_normal_matrix

def get_triangle_mat_from_txt(name):
    mesh_txt = np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\" + name + '.txt')
    
    triangle_matrix = tri_mat_fkt(mesh_txt)
    triangle_norm_mat = calculate_normals(triangle_matrix)
    
    return triangle_matrix, triangle_norm_mat


    


