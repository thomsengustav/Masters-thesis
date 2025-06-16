# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 12:36:29 2024

@author: thoms
"""

'''
plot linser sammen!
'''

import numpy as np
from numba import njit


from Lens_fkt_lib import send_rays_fkt, aspheric_formul, linspace_to_RT_obj, make_lens_obj
from Lens_fkt_lib import ray_tracing, ray_lens_spread, max_def_ang, plot_vectors_fkt, plot_lenses_fkt

def get_spread_rays(N_rays, opt_L, EP, LT, R1, R2, K1, K2):
    y_ls = np.linspace(-3,3, 601)
    z_ls1 = -aspheric_formul(y_ls, R1, K1, 0, 0)
    z_ls2 = aspheric_formul(y_ls, R2, K2, 0, 0)
    
    ls_center2 = np.array([LT,0])
    ls_center1 = np.array([0,0])
    plane_x = opt_L + EP
    
    Lines_obj_mat1 = linspace_to_RT_obj(z_ls1, y_ls, ls_center1)
    Lines_obj_mat2 = linspace_to_RT_obj(z_ls2, y_ls, ls_center2)
    Lens_obj_mat = make_lens_obj(Lines_obj_mat1, Lines_obj_mat2, 1.5245)
    
    origin = np.array([EP, 0])
    ray_mat, org_mat, int_mat = send_rays_fkt(origin, 7.5, N_rays)
    int_mat_n, def_ang_mat = ray_tracing(ray_mat, org_mat, int_mat, Lens_obj_mat, plane_x)
    
    ray_mat = np.zeros((2,N_rays))
    org_mat = np.zeros((2,N_rays))
    for i in range(N_rays):
        ray = int_mat_n[:,i,3] - int_mat_n[:,i,2]
        norm_ray = ray / np.linalg.norm(ray)
        org_mat[:,i] = int_mat_n[:,i,2]
        ray_mat[:,i] = norm_ray
    return ray_mat, org_mat

def get_parallel_Value(int_mat_n2):
    N = int(len(int_mat_n2[0,:,0]))
    y_tot = 0
    for i in range(N):
        vec = int_mat_n2[:,i,-1] - int_mat_n2[:,i,-2]
        vec_norm = vec / np.linalg.norm(vec)
        #print(vec_norm)
        y_tot += abs(vec_norm[1])
        #print(y_tot)
    y_avg = y_tot / N
    return y_avg

def plot_lenses(N_rays, em_d, LTa, R1a, R2a, K1a, K2a, 
                LTb, R1b, R2b, K1b, K2b, d):
    
    # plot_best_lens(N_rays, opt_L, EP, LT, R1, R2, K1, K2):
    y_ls = np.linspace(-3,3, 1001)
    z_ls1 = -aspheric_formul(y_ls, R1a, K1a, 0, 0)
    z_ls2 = aspheric_formul(y_ls, R2a, K2a, 0, 0)
    
    ls_center2 = np.array([LTa,0])
    ls_center1 = np.array([0,0])
    plane_x = 5
    
    Lines_obj_mat1 = linspace_to_RT_obj(z_ls1, y_ls, ls_center1)
    Lines_obj_mat2 = linspace_to_RT_obj(z_ls2, y_ls, ls_center2)
    Lens_obj_mat = make_lens_obj(Lines_obj_mat1, Lines_obj_mat2, 1.5245)
    
    origin = np.array([-10-d, 0])
    ray_mat, org_mat, int_mat = send_rays_fkt(origin, 7.5, N_rays)
    int_mat_n, def_ang_mat = ray_tracing(ray_mat, org_mat, int_mat, Lens_obj_mat, plane_x)
    plot_vectors_fkt(int_mat_n, Lens_obj_mat)
    
    ray_mat_spread, org_mat_spread = get_spread_rays(N_rays, 20, -10-d, LTa, R1a, R2a, K1a, K2a)
        
    y_ls = np.linspace(-5.5,5.5, 1401)
    z_ls1 = aspheric_formul(y_ls, R1b, K1b, 0, 0)
    z_ls2 = -aspheric_formul(y_ls, R2b, K2b, 0, 0)
    em_start_x = org_mat_spread[0, int(N_rays/2)]
    org_mat_spread[0,:] = org_mat_spread[0,:] - em_start_x
    
    ls_center1 = np.array([em_d,0])
    ls_center2 = np.array([em_d+LTb,0])
    plane_x = 200
    
    Lines_obj_mat1 = linspace_to_RT_obj(z_ls1, y_ls, ls_center1)
    Lines_obj_mat2 = linspace_to_RT_obj(z_ls2, y_ls, ls_center2)
    Lens_obj_mat2 = make_lens_obj(Lines_obj_mat1, Lines_obj_mat2, 1.5245)
    
    int_mat = np.zeros((2,N_rays,4))
    int_mat[0,:,0] = org_mat_spread[0,:]
    int_mat[1,:,0] = org_mat_spread[1,:]
    int_mat_n2, def_ang_mat = ray_tracing(ray_mat_spread, org_mat_spread, int_mat, Lens_obj_mat2, plane_x)
    plot_vectors_fkt(int_mat_n2, Lens_obj_mat2)
    
    Lines_obj_mat1 = linspace_to_RT_obj(z_ls1+LTa, y_ls, ls_center1)
    Lines_obj_mat2 = linspace_to_RT_obj(z_ls2+LTa, y_ls, ls_center2)
    Lens_obj_mat2 = make_lens_obj(Lines_obj_mat1, Lines_obj_mat2, 1.5245)
    int_mat_n2[0,:,:] = int_mat_n2[0,:,:] + LTa
    plot_lenses_fkt(int_mat_n, int_mat_n2, Lens_obj_mat, Lens_obj_mat2)
    y_AVG = get_parallel_Value(int_mat_n2)
    print(np.sin(y_AVG)*180/np.pi)

# spherical lenses
# plot_lenses(20, 9, 1, 5, 500, 0, 0, 
#                 2.8, 500, 9.4, 0, 0, 0)

plot_lenses(20, 2.9, 1.57, 4, 22.8, 0, -4, 
                3.5, 9, 8.85, -3, -2.1, 3.2)

y = 5.5
r = 9.4
print(np.sqrt(r**2 - y**2))