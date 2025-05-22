# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 10:04:37 2025

@author: thoms
"""

'''
split triangles and other related functions
'''

import numpy as np
from fast_vec_operators import len_fkt
from BVH_library_funcs import calculate_centroid_fkt

# input cen_tri_node_mat (matrix with the xzy positions of ever vertec and the triangle centroid)
# input NTri_in_leafs_vec (vector of the number of triangles in each leaf of the BVH)
# function working on BVH data
def cal_large_tri_dist(cen_tri_node_mat, NTri_in_leafs_vec):
    # setup
    tri_tot = int(np.sum(NTri_in_leafs_vec))
    dist_vertex_cen_mat = np.zeros((3,tri_tot))
    leafs = int(len(cen_tri_node_mat[0,0,0,:]))
    # loop over every triangle to calculate the distnce from every vertex to the triangle centroid
    counter = 0
    for i in range(leafs):
        for n in range(int(NTri_in_leafs_vec[i])):
            V1_cen_vec = cen_tri_node_mat[0,:,n,i] - cen_tri_node_mat[3,:,n,i]
            V2_cen_vec = cen_tri_node_mat[1,:,n,i] - cen_tri_node_mat[3,:,n,i]
            V3_cen_vec = cen_tri_node_mat[2,:,n,i] - cen_tri_node_mat[3,:,n,i]
            dist_vertex_cen_mat[0,counter] = len_fkt(V1_cen_vec)
            dist_vertex_cen_mat[1,counter] = len_fkt(V2_cen_vec)
            dist_vertex_cen_mat[2,counter] = len_fkt(V3_cen_vec)
            counter += 1
    return dist_vertex_cen_mat

# function working on mesh before BVH
def cal_large_tri_dist_preBVH(centroid_triangle_mat):
    N_tri = int(len(centroid_triangle_mat[0,0,:]))
    V1_dist_vec = np.zeros(N_tri)
    V2_dist_vec = np.zeros(N_tri)
    V3_dist_vec = np.zeros(N_tri)
    for i in range(N_tri):
        V1_distV = centroid_triangle_mat[0,:,i] - centroid_triangle_mat[3,:,i]
        V2_distV = centroid_triangle_mat[1,:,i] - centroid_triangle_mat[3,:,i]
        V3_distV = centroid_triangle_mat[2,:,i] - centroid_triangle_mat[3,:,i]
        V1_dist_vec[i] = len_fkt(V1_distV)
        V2_dist_vec[i] = len_fkt(V2_distV)
        V3_dist_vec[i] = len_fkt(V3_distV)
        
    return V1_dist_vec, V2_dist_vec, V3_dist_vec

# function for finding the largest cen-tri distance
# and returning the index and size
def get_max_D_from3(D1, D2, D3):
    D_largets = np.max(np.array([D1, D2, D3]))
    
    if D_largets == D1:
        return D_largets, 0
    elif D_largets == D2:
        return D_largets, 1
    elif D_largets == D3:
        return D_largets, 2
    
# return D_largest and triangle index in matrix form 
def get_D_largets_mat(V1_dist_vec, V2_dist_vec, V3_dist_vec):
    N = int(len(V1_dist_vec))
    D_largest_mat = np.zeros((N,2))
    for i in range(N):
        D1 = V1_dist_vec[i]
        D2 = V2_dist_vec[i]
        D3 = V3_dist_vec[i]
        
        D_largets, v_index = get_max_D_from3(D1, D2, D3) 
        D_largest_mat[i,:] = D_largets, v_index
    return D_largest_mat


# function for checking the size of D_largets against the larges allowed cen-tri value D_max 
def get_tri_ver_index(D_max, D_largest_mat):
    N = int(len(D_largest_mat[:,0]))
    
    Tri_index_vec = np.zeros(N)
    V_index_vec = np.zeros(N)
    counter = 0
    for i in range(N):
        D_large = D_largest_mat[i,0]
        if D_large > D_max:
            Tri_index_vec[counter] =  i
            V_index_vec[counter] = D_largest_mat[i,1]
            counter +=1
    Tri_index_vec_n = Tri_index_vec[:counter]
    V_index_vec_n = V_index_vec[:counter]
    return Tri_index_vec_n, V_index_vec_n


# find the triangle vertex closest to the vertex with D_larges
def get_closest_vertex(Tri_index_vec, V_index_vec, centroid_triangle_mat):
    N = int(len(Tri_index_vec))
    close_vec = np.zeros(N)
    
    for i in range(N):
        V_index = int(V_index_vec[i])
        tri_index = int(Tri_index_vec[i])
        Vertex = centroid_triangle_mat[V_index,:,tri_index]
        
        dist1 = len_fkt(centroid_triangle_mat[(V_index +1)%3 ,:,tri_index]-Vertex)
        dist2 = len_fkt(centroid_triangle_mat[(V_index +2)%3 ,:,tri_index]-Vertex)
        
        if dist1 < dist2:
            close_vec[i] = 0 
        else:
            close_vec[i] = 1
    return close_vec

def split_large_tri(Triangle_matrix, max_dist):
    # calculate centroid mat
    centroid_triangle_mat = calculate_centroid_fkt(Triangle_matrix)
    # setup for finding triangles to split and getting the corret triangle indexes
    V1_dist_vec, V2_dist_vec, V3_dist_vec = cal_large_tri_dist_preBVH(centroid_triangle_mat)
    D_largest_mat = get_D_largets_mat(V1_dist_vec, V2_dist_vec, V3_dist_vec)
    Tri_index_vec_n, V_index_vec_n = get_tri_ver_index(max_dist, D_largest_mat)
    close_vec = get_closest_vertex(Tri_index_vec_n, V_index_vec_n, centroid_triangle_mat)
    
    # perform splitting algorithm
    N_large_tri = int(len(V_index_vec_n))
    extra_tri_mat = np.zeros((3,3,2*N_large_tri))
    N_tri = int(len(Triangle_matrix[0,0,:]))
    n_Triangle_matrix = np.zeros((3,3,N_tri))
    n_Triangle_matrix[:,:,:] = Triangle_matrix[:,:,:]
    for i in range(N_large_tri):
        tri_index = int(Tri_index_vec_n[i])
        vertex_far = int(V_index_vec_n[i])
        V11 = Triangle_matrix[(vertex_far) % 3,:,tri_index]
        V22 = Triangle_matrix[(vertex_far+1) % 3,:,tri_index]
        V33 = Triangle_matrix[(vertex_far+2) % 3,:,tri_index]
        if close_vec[i] == 0:
            V1, V2, V3 = V11, V22, V33 
        else:
            V1, V3, V2 = V11, V22, V33 
        V4 = (V1+V2)/2 
        V5 = (V1+V3)/2 
        
        # by splitting we create 3 triangles from 1
        # save first triangle
        n_Triangle_matrix[0,:,tri_index] = V1
        n_Triangle_matrix[1,:,tri_index] = V4
        n_Triangle_matrix[2,:,tri_index] = V5

        #add 2 new triangles to extra_tri_mat
        extra_tri_mat[0,:,i*2] = V2
        extra_tri_mat[1,:,i*2] = V5
        extra_tri_mat[2,:,i*2] = V3
        
        extra_tri_mat[0,:,i*2+1] = V4
        extra_tri_mat[1,:,i*2+1] = V2
        extra_tri_mat[2,:,i*2+1] = V5
    
    New_Triangle_matrix = np.zeros((3,3,N_large_tri*2 + N_tri))
    New_Triangle_matrix[:,:,:N_tri] = n_Triangle_matrix
    New_Triangle_matrix[:,:,N_tri:] = extra_tri_mat
    return New_Triangle_matrix