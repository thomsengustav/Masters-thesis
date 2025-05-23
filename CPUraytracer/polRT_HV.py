# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 14:34:19 2025

@author: thoms
"""

'''
POLRT calculating HH, HV, VH and VV simultaneiously!


'''
import numpy as np

#### Initialize  parameters!

pol_name = "tank_pol_2" # name for saving data

###
radar_range = 100 # meters - 0
slant_angle = 25 # degrees - 1

azimuth_start = -5 # degrees - 2
azimuth_end = 5 # degrees - 3
azimuth_steps = 120 # - 4

max_parameter_t = 150 # longest allowed ray in meters - 5
max_in_leaf = 300 # - 6
max_dist = 0.3 # - 7

n_index = 300 # - 8

N_rays = 1800000 # number of primary rays launched from source # - 9
N_bounces = 3 # number of allowed bounces # - 10
Detector_radius = 15 # meters # - 11
beam_size = 4 # - 12
ref_size = 5 # - 13
signal_resolution = 0.001 # 1mm # - 14

SETUP_data = (radar_range, slant_angle, azimuth_start, azimuth_end, azimuth_steps, max_parameter_t, max_in_leaf, max_dist,
              n_index, N_rays, N_bounces, Detector_radius, beam_size, ref_size, signal_resolution)

clutter_data = False


#### import geometry
from load_gmesh_square import load_gmesh_triangle_model
from better_mesh_fkt import mesh_setup_fkt
from make_material_matrix import make_material_mat
from rotate_mesh_fkt import rotate_mesh
from load_model_center_rotate import load_big_model

# Triangle_matrix, Triangle_normal_matrix = load_gmesh_triangle_model('m_3_box.txt')
# Triangle_matrix, Triangle_normal_matrix = load_big_model('råsted_vej3') 
Triangle_matrix, Triangle_normal_matrix = load_big_model('t90a_trekanter') 
#Triangle_matrix = Triangle_matrix*200 # råsted_vej

def rotate_ford(Triangle_matrix):
    N = int(len(Triangle_matrix[0,0,:]))
    new_Triangle_matrix = np.zeros((3,3,N))
    new_Triangle_matrix[:,0,:] = Triangle_matrix[:,0,:]
    new_Triangle_matrix[:,1,:] = Triangle_matrix[:,2,:]
    new_Triangle_matrix[:,2,:] = Triangle_matrix[:,1,:]
    return new_Triangle_matrix

# Triangle_matrix = rotate_ford(Triangle_matrix)

Triangle_matrix, Triangle_normal_matrix = mesh_setup_fkt(Triangle_matrix, max_dist)
Triangle_matrix = rotate_mesh(Triangle_matrix, 35)

Material_matrix = make_material_mat(Triangle_matrix, n_index, clutter_data)


from BVH import plot_bounding_volume, find_tri_in_leafs, tri_normal_calculator_BVH_fkt, tree_node_fkt, optimize_index_array
from centroid_on_rasterization import BVH_builder_materials
### setup BVH!
box_node_mat, cen_tri_node_mat, parent_vec, N_box_vec, material_mat_node = BVH_builder_materials(Triangle_matrix, Material_matrix, max_in_leaf)

index_Array, N_diff_vec = tree_node_fkt(box_node_mat, cen_tri_node_mat, N_box_vec, parent_vec)
new_index_array = optimize_index_array(index_Array, box_node_mat, cen_tri_node_mat, N_diff_vec)
NTri_in_leafs_vec = find_tri_in_leafs(cen_tri_node_mat)
Triangle_normal_matrix = tri_normal_calculator_BVH_fkt(cen_tri_node_mat)

plot_bounding_volume(new_index_array, box_node_mat, cen_tri_node_mat, N_diff_vec, np.array([-5, 5]), np.array([-5, 5]), np.array([0, 10]), 30, 30, 0)

BVH_data = (Triangle_normal_matrix, material_mat_node, new_index_array, box_node_mat, cen_tri_node_mat, NTri_in_leafs_vec)

from pol_int_HV import SAR_pol_HV

# run main polRT function!
SAR_pol_HV(pol_name, SETUP_data, BVH_data)