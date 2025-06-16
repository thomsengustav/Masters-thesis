# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 14:04:29 2024

@author: thoms
"""

'''
Better sweep for lenses!
'''

import numpy as np
from numba import njit, prange
import time
# Mat_par - start, end, len for all paramters


N_pars = 6
Mat_par = np.zeros((3,N_pars))

# emitpos_con = make_condition(-11, -5, 6)
# LensT_con = make_condition(0.5, 3, 8)
# R1_con = make_condition(4, 25, 20)
# R2_con = make_condition(4, 25, 20)
# k1_con = make_condition(-4, 2, 15)
# k2_con = make_condition(-4, 2, 15)

constanc_vec = np.zeros(4)
constanc_vec[0] = 1.5245 # refractive index
constanc_vec[1] = 20 # max optical length
constanc_vec[2] = 10 # N_rays 
constanc_vec[3] = 7.5 # emitter 

Mat_par[:,0] = np.array([-15, -5, 12]) # emitter position
Mat_par[:,1] = np.array([0.5, 4, 10]) # lens thickness
Mat_par[:,2] = np.array([4, 25, 20]) # R1
Mat_par[:,3] = np.array([4, 25, 20]) # R2
Mat_par[:,4] = np.array([0, 0, 1]) # k1
Mat_par[:,5] = np.array([0, 0, 1]) # k2

# fourth1_con =  make_condition(-0.0, 0.0, 1) # 0.01
# fourth2_con =  make_condition(-0.0, 0.0, 1)
# sixth1_con =  make_condition(-0.00, 0.00, 1) # 0.005
# sixth2_con =  make_condition(-0.00, 0.00, 1)


# sweep_mat_par # N_sweep x M_res/par matrix
@njit
def make_sweep_mat_par(Mat_par, N_par):
    
    N_sweep = 1
    N_vec = np.zeros(N_par)
    for i in range(N_par):
        N_sweep = N_sweep * Mat_par[2,i]
        N_vec[i] =  Mat_par[2,i]
    
    N_mul_vec = np.zeros(N_par)
    N_mul_vec[0:] = N_vec[0]
    for i in range(N_par-1):
        N_mul_vec[i+1:] *= N_vec[i+1]
    
    sweep_mat_par = np.zeros((int(N_par), int(N_sweep)))

    ls0 = np.linspace(Mat_par[0,0], Mat_par[1,0], int(N_vec[0]))
    ls1 = np.linspace(Mat_par[0,1], Mat_par[1,1], int(N_vec[1]))
    ls2 = np.linspace(Mat_par[0,2], Mat_par[1,2], int(N_vec[2]))
    ls3 = np.linspace(Mat_par[0,3], Mat_par[1,3], int(N_vec[3]))
    ls4 = np.linspace(Mat_par[0,4], Mat_par[1,4], int(N_vec[4]))
    ls5 = np.linspace(Mat_par[0,5], Mat_par[1,5], int(N_vec[5]))
    
    print(N_mul_vec[5])
    counter_1 = -1
    counter_2 = -1
    counter_3 = -1
    counter_4 = -1
    counter_5 = -1
    
    for i in range(int(N_sweep)):
        
        counter_0 = i % N_vec[0]
        mod1 = i % N_mul_vec[0]
        mod2 = i % N_mul_vec[1]
        mod3 = i % N_mul_vec[2]
        mod4 = i % N_mul_vec[3]
        mod5 = i % N_mul_vec[4]
        
        if mod1 == 0:
            counter_1 += 1 
            counter_1 = counter_1 % N_vec[1]
        if mod2 == 0:
            counter_2 += 1 
            counter_2 = counter_2 % N_vec[2]
        if mod3 == 0:
            counter_3 += 1 
            counter_3 = counter_3 % N_vec[3]
        if mod4 == 0:
            counter_4 += 1 
            counter_4 = counter_4 % N_vec[4]
        if mod5 == 0:
            counter_5 += 1 
        
        sweep_mat_par[0,i] = ls0[int(counter_0)]
        sweep_mat_par[1,i] = ls1[int(counter_1)]
        sweep_mat_par[2,i] = ls2[int(counter_2)]
        sweep_mat_par[3,i] = ls3[int(counter_3)]
        sweep_mat_par[4,i] = ls4[int(counter_4)]
        sweep_mat_par[5,i] = ls5[int(counter_5)]
    
    return sweep_mat_par, N_mul_vec

# t0 = time.time()
# test = make_sweep_mat_par(Mat_par, 6)
# print(time.time()-t0)


from Lens_fkt_lib import send_rays_fkt, aspheric_formul, linspace_to_RT_obj, make_lens_obj
from Lens_fkt_lib import ray_tracing, ray_lens_spread, max_def_ang, plot_vectors_fkt

@njit(parallel=True)
def sweep_fkt(Mat_par, constanc_vec):
    sweep_mat_par, N_mul_vec = make_sweep_mat_par(Mat_par, 6)
    
    N_sweep = int(N_mul_vec[5])
    Sweep_res = np.zeros((8,N_sweep))
    N_rays = int(constanc_vec[2])
    OL = constanc_vec[1]
    Spred = constanc_vec[3]
    nL = constanc_vec[0]
    
    y_ls = np.linspace(-3,3, 601)
    ls_center1 = np.array([0,0])
    for i in prange(N_sweep):
        sweep_vec = sweep_mat_par[:,i]
        EP = sweep_vec[0]
        LT = sweep_vec[1]
        R1 = sweep_vec[2]
        R2 = sweep_vec[3]
        K1 = sweep_vec[4]
        K2 = sweep_vec[5]
        
        origin = np.array([EP, 0])
        ray_mat, org_mat, int_mat = send_rays_fkt(origin, Spred, N_rays)
        plane_x = OL + EP
        
        ls_center2 = np.array([LT,0])
        
        z_ls1 = -aspheric_formul(y_ls, R1, K1, 0, 0)
        z_ls2 = aspheric_formul(y_ls, R2, K2, 0, 0)
        
        Lines_obj_mat1 = linspace_to_RT_obj(z_ls1, y_ls, ls_center1)
        Lines_obj_mat2 = linspace_to_RT_obj(z_ls2, y_ls, ls_center2)
        Lens_obj_mat = make_lens_obj(Lines_obj_mat1, Lines_obj_mat2, nL)
        int_mat_n, def_ang_mat = ray_tracing(ray_mat, org_mat, int_mat, Lens_obj_mat, plane_x)
 
        res_Vec = np.zeros(8)
        res_Vec[:6] = sweep_vec
        res_Vec[6] = ray_lens_spread(int_mat_n)
        res_Vec[7] = max_def_ang(def_ang_mat, N_rays)
        
        Sweep_res[:,i] = res_Vec
    return Sweep_res

# t0 = time.time()
# res = sweep_fkt(Mat_par, constanc_vec)
# print(time.time()-t0)  
# res = np.load('lens_search_res.npy') 







def remove_bad_spread(Sweep_res, min_spread):
    N_sweep = len(Sweep_res[0,:])
    new_mat = np.zeros((8,N_sweep))
    counter = 0
    for i in range(N_sweep):
        if Sweep_res[6,i] >= min_spread:
            new_mat[:,counter] = Sweep_res[:,i]
            counter += 1
    
    filtered_mat = new_mat[:,:counter]
    return filtered_mat



def plot_best_properties(Sweep_res, N, min_spread):
    
    res_mat = remove_bad_spread(Sweep_res, min_spread)
    N_res = len(res_mat[0,:])
    sorted_mat = res_mat[:,res_mat[7,:].argsort()]
    
    best_set = sorted_mat[:,N]
    return sorted_mat

mat = plot_best_properties(res, 10, 8)

def save_counts_values(counts, values, Mat):
    M = int(len(counts))
    if M > len(Mat[0,:]):
        m_len = int(len(Mat[:,0])) + 2
        m_row = int(len(Mat[0,:]))
        Mat_n = np.ones((m_len, M))*(-999)
        Mat_n[:-2, :m_row] = Mat
        Mat = Mat_n
        Mat[-2, :] = values
        Mat[-1,:] = counts
    else:
        m_row = int(len(Mat[0,:]))
        m_len = int(len(Mat[:,0])) + 2
        Mat_n = np.ones((m_len, m_row))*(-999)
        Mat_n[:-2,:] = Mat
        Mat = Mat_n
        Mat[-2,:M] = values
        Mat[-1,:M] = counts
    return Mat
        

def count_parameters(best_set, max_def_ang):
    ang_vec = best_set[-1,:]
    set_ang_vec = ang_vec[np.where(ang_vec <= max_def_ang)]
    N = int(len(set_ang_vec))
    
    Mat = np.zeros((1,1))
    for i in range(6):
        values = np.array([])
        counts = np.array([])
        for n in range(N):
            var = best_set[i,n]
            if var not in values:
                values = np.append(values, var)
                counts_n = np.zeros(int(len(values)))
                counts_n[:-1] = counts 
                counts = counts_n
            index = np.where(values == var)
            counts[int(index[0])] += 1
        Mat = save_counts_values(counts, values, Mat)
    return Mat
        
        
par_mat = count_parameters(mat, 0.01)


def plot_best_lens(N_rays, opt_L, EP, LT, R1, R2, K1, K2):
    y_ls = np.linspace(-3,3, 601)
    z_ls1 = -aspheric_formul(y_ls, R1, K1, 0, 0)
    z_ls2 = aspheric_formul(y_ls, R2, K2, 0, 0)
    
    ls_center2 = np.array([LT,0])
    ls_center1 = np.array([0,0])
    plane_x = opt_L + EP
    
    Lines_obj_mat1 = linspace_to_RT_obj(z_ls1, y_ls, ls_center1)
    Lines_obj_mat2 = linspace_to_RT_obj(z_ls2, y_ls, ls_center2)
    Lens_obj_mat = make_lens_obj(Lines_obj_mat1, Lines_obj_mat2, 1.5245)
    
    emitpos = EP
    origin = np.array([EP, 0])
    ray_mat, org_mat, int_mat = send_rays_fkt(origin, 7.5, 10)
    int_mat_n, def_ang_mat = ray_tracing(ray_mat, org_mat, int_mat, Lens_obj_mat, plane_x)
    plot_vectors_fkt(int_mat_n, Lens_obj_mat)


plot_best_lens(20, 20, -13.2, 1.277, 4, 23.9, 0, 0)

LT = 1.571
R1, R2 = 4, 22.8
K1, K2 = 0, -4

y_ls = np.linspace(-3,3, 601)
ls_center1 = np.array([0,0])
ls_center2 = np.array([LT,0])

z_ls1 = -aspheric_formul(y_ls, R1, K1, 0, 0)
z_ls2 = aspheric_formul(y_ls, R2, K2, 0, 0)

Lines_obj_mat1 = linspace_to_RT_obj(z_ls1, y_ls, ls_center1)
Lines_obj_mat2 = linspace_to_RT_obj(z_ls2, y_ls, ls_center2)

x1 = Lines_obj_mat1[5,0]
x2 = Lines_obj_mat2[5,0]

Len = x2 - x1
print(Len)


'''
kollimerings linse!
- regn grad af parralelle rays
- regn hvor konsitent den er
'''

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

ray_mat_spread, org_mat_spread = get_spread_rays(10, 20, -13.2, 1.277, 4, 23.9, 0, 0)

import matplotlib.pyplot as plt

ax = plt.figure().add_subplot()
for i in range(int(len(org_mat_spread[0,:]))):
    ax.plot([1.277+org_mat_spread[0,i], 1.277+org_mat_spread[0,i]+ray_mat_spread[0,i]*6.8], [org_mat_spread[1,i], org_mat_spread[1,i]+ray_mat_spread[1,i]*6.8],'r')

ax.set_xlim(-5, 100)
ax.set_ylim(-6, 6)
ax.set_xlabel('X')
ax.set_ylabel('Y')
plt.show()


constanc_koll_vec = np.zeros(3)
constanc_koll_vec[0] = 1.5245 # refractive index
constanc_koll_vec[1] = 15 # max optical length
constanc_koll_vec[2] = 10 # N_rays 

Mat_koll_par = np.zeros((3,6))
Mat_koll_par[:,0] = np.array([6, 8, 10]) # distance from spread
Mat_koll_par[:,1] = np.array([1.5, 4, 12]) # lens thickness
Mat_koll_par[:,2] = np.array([6, 40, 30]) # R1
Mat_koll_par[:,3] = np.array([6, 40, 30]) # R2
Mat_koll_par[:,4] = np.array([0, 0, 1]) # k1
Mat_koll_par[:,5] = np.array([0, 0, 1]) # k2



@njit
def make_sweep_mat_koll_par(Mat_koll_par, N_par):
    N_sweep = 1
    N_vec = np.zeros(N_par)
    for i in range(N_par):
        N_sweep = N_sweep * Mat_koll_par[2,i]
        N_vec[i] =  Mat_par[2,i]
    
    N_mul_vec = np.zeros(N_par)
    N_mul_vec[0:] = N_vec[0]
    for i in range(N_par-1):
        N_mul_vec[i+1:] *= N_vec[i+1]
    
    sweep_mat_par = np.zeros((int(N_par), int(N_sweep)))

    ls0 = np.linspace(Mat_par[0,0], Mat_par[1,0], int(N_vec[0]))
    ls1 = np.linspace(Mat_par[0,1], Mat_par[1,1], int(N_vec[1]))
    ls2 = np.linspace(Mat_par[0,2], Mat_par[1,2], int(N_vec[2]))
    ls3 = np.linspace(Mat_par[0,3], Mat_par[1,3], int(N_vec[3]))
    ls4 = np.linspace(Mat_par[0,4], Mat_par[1,4], int(N_vec[4]))
    
    print(N_mul_vec[4])
    counter_1 = -1
    counter_2 = -1
    counter_3 = -1
    counter_4 = -1
    
    for i in range(int(N_sweep)):
        
        counter_0 = i % N_vec[0]
        mod1 = i % N_mul_vec[0]
        mod2 = i % N_mul_vec[1]
        mod3 = i % N_mul_vec[2]
        mod4 = i % N_mul_vec[3]
        
        if mod1 == 0:
            counter_1 += 1 
            counter_1 = counter_1 % N_vec[1]
        if mod2 == 0:
            counter_2 += 1 
            counter_2 = counter_2 % N_vec[2]
        if mod3 == 0:
            counter_3 += 1 
            counter_3 = counter_3 % N_vec[3]
        if mod4 == 0:
            counter_4 += 1 
        
        sweep_mat_par[0,i] = ls0[int(counter_0)]
        sweep_mat_par[1,i] = ls1[int(counter_1)]
        sweep_mat_par[2,i] = ls2[int(counter_2)]
        sweep_mat_par[3,i] = ls3[int(counter_3)]
        sweep_mat_par[4,i] = ls4[int(counter_4)]
    
    return sweep_mat_par, N_mul_vec

sweep_mat_koll_par, N_koll_mul_vec = make_sweep_mat_par(Mat_koll_par, 6)

@njit
def how_koll_fkt(int_mat_n):
    N = int(len(int_mat_n[0,:,0]))
    parallel_var = 0.000001
    for i in range(N):
        ray = int_mat_n[:,i,3] - int_mat_n[:,i,2]
        norm_ray = ray / np.linalg.norm(ray)
        
        if abs(norm_ray[1]) == np.nan:
            print('nan')
            return 99
        if abs(norm_ray[1]) > parallel_var:
            parallel_var = abs(norm_ray[1])
    theta = np.arcsin(parallel_var)*180/np.pi
    return theta
        
@njit
def radius_on_koll_lens(int_mat_n):
    max_h = np.max(int_mat_n[1,:,2])
    R = max_h*2
    return R

@njit(parallel=True)
def sweep_koll_fkt(Mat_koll_par, constanc_koll_vec, ray_mat_spread, org_mat_spread):
    sweep_mat_par, N_mul_vec = make_sweep_mat_par(Mat_koll_par, 6)
    N_sweep = int(N_mul_vec[5])
    Sweep_res = np.zeros((8,N_sweep))
    
    N_rays = int(constanc_vec[2])
    OL = constanc_vec[1]
    nL = constanc_vec[0]
    
    y_ls = np.linspace(-7,7, 1191)
    
    em_start_x = org_mat_spread[0, int(N_rays/2)]
    org_mat_spread[0,:] = org_mat_spread[0,:] - em_start_x
    print(N_sweep)
    for i in prange(N_sweep):
        sweep_vec = sweep_mat_par[:,i]
        em_d = sweep_vec[0]
        LT = sweep_vec[1]
        R1 = sweep_vec[2]
        R2 = sweep_vec[3]
        K1 = sweep_vec[4]
        K2 = sweep_vec[5]
        
        plane_x = OL + em_start_x
        
        ls_center1 = np.array([em_d,0])
        ls_center2 = np.array([em_d+LT,0])
        int_mat = np.zeros((2,N_rays,4))
        int_mat[0,:,0] = org_mat_spread[0,:]
        int_mat[1,:,0] = org_mat_spread[1,:]
        
        z_ls1 = aspheric_formul(y_ls, R1, K1, 0, 0)
        z_ls2 = -aspheric_formul(y_ls, R2, K2, 0, 0)
        
        Lines_obj_mat1 = linspace_to_RT_obj(z_ls1, y_ls, ls_center1)
        Lines_obj_mat2 = linspace_to_RT_obj(z_ls2, y_ls, ls_center2)
        Lens_obj_mat = make_lens_obj(Lines_obj_mat1, Lines_obj_mat2, nL)
        int_mat_n, def_ang_mat = ray_tracing(ray_mat_spread, org_mat_spread, int_mat, Lens_obj_mat, plane_x)
        
        res_Vec = np.zeros(8)
        res_Vec[:6] = sweep_vec
        res_Vec[6] = how_koll_fkt(int_mat_n)
        res_Vec[7] = max_def_ang(def_ang_mat, N_rays)
        
        Sweep_res[:,i] = res_Vec
    return Sweep_res

t0 = time.time()
ressss = sweep_koll_fkt(Mat_koll_par, constanc_koll_vec, ray_mat_spread, org_mat_spread)
print(time.time()-t0)

def remove_bad_spread_koll(Sweep_res, min_spread):
    N_sweep = len(Sweep_res[0,:])
    new_mat = np.zeros((8,N_sweep))
    counter = 0
    for i in range(N_sweep):
        if Sweep_res[6,i] <= min_spread:
            new_mat[:,counter] = Sweep_res[:,i]
            counter += 1
    
    filtered_mat = new_mat[:,:counter]
    return filtered_mat


res_minus = remove_bad_spread_koll(ressss, 0.5)
#print(ressss[5,300000])
def plot_best_properties_koll(Sweep_res, N, min_spread):
    
    res_mat = remove_bad_spread_koll(Sweep_res, min_spread)
    sorted_mat = res_mat[:,res_mat[7,:].argsort()]
    
    best_set = sorted_mat[:,N]
    return sorted_mat
res_sort = plot_best_properties_koll(res_minus, 100, 0.4)

def plot_best_lens_koll(N_rays, opt_L, em_d, LT, R1, R2, K1, K2, org_mat_spread, ray_mat_spread):
    y_ls = np.linspace(-6,6, 601)
    z_ls1 = aspheric_formul(y_ls, R1, K1, 0, 0)
    z_ls2 = -aspheric_formul(y_ls, R2, K2, 0, 0)
    em_start_x = org_mat_spread[0, int(N_rays/2)]
    org_mat_spread[0,:] = org_mat_spread[0,:] - em_start_x
    
    ls_center1 = np.array([em_d,0])
    ls_center2 = np.array([em_d+LT,0])
    plane_x = opt_L + em_start_x
    
    Lines_obj_mat1 = linspace_to_RT_obj(z_ls1, y_ls, ls_center1)
    Lines_obj_mat2 = linspace_to_RT_obj(z_ls2, y_ls, ls_center2)
    Lens_obj_mat = make_lens_obj(Lines_obj_mat1, Lines_obj_mat2, 1.5245)
    
    int_mat = np.zeros((2,N_rays,4))
    int_mat[0,:,0] = org_mat_spread[0,:]
    int_mat[1,:,0] = org_mat_spread[1,:]
    int_mat_n, def_ang_mat = ray_tracing(ray_mat_spread, org_mat_spread, int_mat, Lens_obj_mat, plane_x)
    plot_vectors_fkt(int_mat_n, Lens_obj_mat)

plot_best_lens_koll(10, 80, 6.66, 2.4, 30.6, 9.5, 0, -0, org_mat_spread, ray_mat_spread)






