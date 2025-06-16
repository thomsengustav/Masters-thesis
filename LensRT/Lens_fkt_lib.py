# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:37:52 2024

@author: thoms
"""

# function library of lenses

import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt

@njit
def send_rays_fkt(origin, spread, N):
    max_height = np.tan(spread/180*np.pi)
    spread_vec = np.linspace(max_height, -max_height, N)
    vec_mat = np.zeros((2,N))
    origin_mat = np.zeros((2,N))
    origin_mat[0,:] = origin[0]
    origin_mat[1,:] = origin[1]
    intersection_mat = np.zeros((2,N,4))
    intersection_mat[0,:,0] = origin[0]
    intersection_mat[1,:,0] = origin[1]
    for i in range(N):
        x = (origin[0]+1) - origin[0]
        y = spread_vec[i] - origin[1]
        ray_vec = np.array([x,y])
        ray_vec_norm = ray_vec / np.linalg.norm(ray_vec)
        vec_mat[:,i] = ray_vec_norm
    return vec_mat, origin_mat, intersection_mat

@njit
def aspheric_formul(r, R, k, par1, par2):
    div = R * (1+np.sqrt(1-(1+k)*r**2 / (R**2)))
    z = r**2 / div + par1 * r**4 + par2 * r**6
    return z

@njit
def get_line_ab(linex, liney):
    a = (liney[1]- liney[0]) / (linex[1]- linex[0])
    b = liney[0] - a * linex[0]
    return a, b

@njit
def get_line_normal(linex, liney):
    line_vec = np.array([linex[1] - linex[0], liney[1] - liney[0]])
    line_vec = line_vec / np.linalg.norm(line_vec)
    
    norm_vec = np.array([-line_vec[1], line_vec[0]])
    
    # always pick norm_vec to 'point' left
    check_vec = np.array([1,0])
    dot_prod = check_vec[0]*norm_vec[0] + check_vec[1]*norm_vec[1]
    
    if dot_prod > 0:
        norm_vec = np.array([line_vec[1], -line_vec[0]])
        return norm_vec
    else:
        return norm_vec

@njit
def linspace_to_RT_obj(z_ls, y_ls, ls_center): 
    N_segs = int(len(z_ls)) - 1
    # obj_mat to store a, b, linex, liney and norm_vec
    Lines_obj_mat = np.zeros((8,N_segs))
    # move surface to correct position
    z_ls_lens = z_ls + ls_center[0]
    y_ls_lens = y_ls + ls_center[1]
    for i in range(N_segs):
        linex, liney = np.array([z_ls_lens[i], z_ls_lens[1+i]]), np.array([y_ls_lens[i], y_ls_lens[1+i]])
        Lines_obj_mat[0, i], Lines_obj_mat[1, i] = get_line_ab(linex, liney)
        Lines_obj_mat[2:4, i] = get_line_normal(linex, liney)
        Lines_obj_mat[4:6, i] = linex
        Lines_obj_mat[6:8, i] = liney
    return Lines_obj_mat

@njit
def get_bot_and_top_point(Lines_obj_mat):
    linex1, linex2 = Lines_obj_mat[4:6, 0], Lines_obj_mat[4:6, -1]
    liney1, liney2 = Lines_obj_mat[6:8, 0], Lines_obj_mat[6:8, -1]
    y_points = np.zeros(4)
    y_points[0:2] = liney1
    y_points[2:4] = liney2
    
    x_points = np.zeros(4)
    x_points[0:2] = linex1
    x_points[2:4] = linex2
    
    min_y = np.min(y_points)
    index_min = int(np.where(y_points == min_y)[0][0])
    
    max_y = np.max(y_points)
    index_max = int(np.where(y_points == max_y)[0][0])
    
    bot_p = np.array([x_points[index_min], min_y])
    top_p = np.array([x_points[index_max], max_y])
    return bot_p, top_p

@njit
def make_lens_obj(Lines_obj_mat1, Lines_obj_mat2, n_lens):
    N1 = int(len(Lines_obj_mat1[0,:]))
    N2 = int(len(Lines_obj_mat2[0,:]))
    
    Lens_obj_mat = np.zeros((9,N1+N2+2))
    Lens_obj_mat[8,:] = n_lens
    
    bot_p1, top_p1 = get_bot_and_top_point(Lines_obj_mat1)
    bot_p2, top_p2 = get_bot_and_top_point(Lines_obj_mat2)
    
    bot_linex, bot_liney = np.array([bot_p1[0], bot_p2[0]]), np.array([bot_p1[1], bot_p2[1]])
    top_linex, top_liney = np.array([top_p1[0], top_p2[0]]), np.array([top_p1[1], top_p2[1]])  
    
    Lens_obj_mat[0, 0], Lens_obj_mat[1, 0] = get_line_ab(bot_linex, bot_liney)
    Lens_obj_mat[2:4, 0] = get_line_normal(bot_linex, bot_liney)
    Lens_obj_mat[4:6, 0] = bot_linex
    Lens_obj_mat[6:8, 0] = bot_liney
    
    Lens_obj_mat[0, 1], Lens_obj_mat[1, 1] = get_line_ab(top_linex, top_liney)
    Lens_obj_mat[2:4, 1] = get_line_normal(top_linex, top_liney)
    Lens_obj_mat[4:6, 1] = top_linex
    Lens_obj_mat[6:8, 1] = top_liney
    
    Lens_obj_mat[0, 2:(N1+2)], Lens_obj_mat[1, 2:(N1+2)] = Lines_obj_mat1[0,:],  Lines_obj_mat1[1,:]
    Lens_obj_mat[2:4, 2:(N1+2)] = Lines_obj_mat1[2:4,:]
    Lens_obj_mat[4:6, 2:(N1+2)] = Lines_obj_mat1[4:6,:]
    Lens_obj_mat[6:8, 2:(N1+2)] = Lines_obj_mat1[6:8,:]
    
    Lens_obj_mat[0, (N1+2):(N1+2+N2)], Lens_obj_mat[1, (N1+2):(N1+2+N2)] = Lines_obj_mat2[0,:],  Lines_obj_mat2[1,:]
    Lens_obj_mat[2:4, (N1+2):(N1+2+N2)] = Lines_obj_mat2[2:4,:]
    Lens_obj_mat[4:6, (N1+2):(N1+2+N2)] = Lines_obj_mat2[4:6,:]
    Lens_obj_mat[6:8, (N1+2):(N1+2+N2)] = Lines_obj_mat2[6:8,:]
    
    return Lens_obj_mat

@njit
def get_deflection_angle(ray_dir, norm_vec):
    def_ang = np.arccos(-ray_dir[0]*norm_vec[0] - ray_dir[1]*norm_vec[1])
    return def_ang

@njit
def hit_plane_fkt(Origin, direction, point, normal):
    dot1 = normal[0] * (point[0]-Origin[0]) + normal[1] * (point[1]-Origin[1])
    dot2 = normal[0] * direction[0] + normal[1] * direction[1]
    t = dot1 / dot2
    if t < 0:
        return False
    else:
        return t

@njit
def snells_law(ray_dir, norm_vec, n0, n1):
    ang0 = np.pi - np.arccos(ray_dir[0]*norm_vec[0] + ray_dir[1]*norm_vec[1])
    ang1 = np.arcsin(n0/n1 * np.sin(ang0))
    
    a_vec = np.array([norm_vec[1], -norm_vec[0]])
    ar_dot = a_vec[0]*ray_dir[0] + a_vec[1]*ray_dir[1]
    
    if ar_dot > 0:
        cos1 = np.cos(ang1)
        sin1 = np.sin(ang1)
        x = -norm_vec[0] * cos1 + norm_vec[1] * sin1
        y = -norm_vec[0] * sin1 - norm_vec[1] * cos1
        new_ray_dir = np.array([x, y])
        new_ray_dir = new_ray_dir / np.linalg.norm(new_ray_dir)
        return new_ray_dir
    else:
        ang1 = -ang1
        cos1 = np.cos(ang1)
        sin1 = np.sin(ang1)
        x = -norm_vec[0] * cos1 + norm_vec[1] * sin1
        y = -norm_vec[0] * sin1 - norm_vec[1] * cos1
        new_ray_dir = np.array([x, y])
        new_ray_dir = new_ray_dir / np.linalg.norm(new_ray_dir)
        return new_ray_dir

@njit
def intercetion_point_fkt(ray_org, ray_dir, t):
    int_point = ray_org + t * ray_dir
    return int_point

@njit
def intersection_in_line(int_point, linex, liney):
    lx1, lx2 = min(linex), max(linex)
    ly1, ly2 = min(liney), max(liney)
    if lx1 <= int_point[0] <= lx2:
        if ly1 <= int_point[1] <= ly2:
            return True
        else:
            return False
    else:
        return False

@njit
def line_ray_int(a, b, ray_dir, ray_org):
    div = ray_dir[1] - a*ray_dir[0]
    if div == 0:
        return False
    t = (a * ray_org[0] + b - ray_org[1]) / div
    return t

@njit
def ray_line_obj_int(ray_dir, ray_org, Lines_obj_mat):
    N_obj = int(len(Lines_obj_mat[0,:]))
    t_hit = 999
    norm_vec = np.array([0., 0])
    for i in range(N_obj):
        a, b = Lines_obj_mat[0, i], Lines_obj_mat[1, i]
        t = line_ray_int(a, b, ray_dir, ray_org)
        if t == False:
            continue
        
        int_point = intercetion_point_fkt(ray_org, ray_dir, t)
        linex, liney = Lines_obj_mat[4:6, i], Lines_obj_mat[6:8, i]
        line_bool = intersection_in_line(int_point, linex, liney)
        
        if line_bool == True:
            if t < t_hit and t > 0.1:
                t_hit = t
                norm_vec = Lines_obj_mat[2:4, i]
    if t_hit == 999:
        return False, norm_vec
    else:
        return t_hit, norm_vec
    
@njit
def max_def_ang(def_ang_mat, N_rays):
    delta_ang_v = np.zeros(int(N_rays * 0.5))
    for i in range(int(N_rays*0.5)):
        delta_ang_v[i] = abs(def_ang_mat[0,i] - def_ang_mat[1,i])
    delta_ang = np.max(delta_ang_v)
    return delta_ang

@njit
def ray_lens_spread(int_mat_n):
    y_vec = int_mat_n[1,:,3]
    max_spread = np.max(y_vec)
    tot_spread = 2 * max_spread
    return tot_spread

@njit
def ray_tracing(ray_mat, org_mat, int_mat, Lens_obj_mat, plane_x):
    
    N = int(len(ray_mat[0,:]))
    #norm_vecs = np.zeros((2,N,2))
    def_ang_mat = np.zeros((2,N))
    for i in range(N):
        ray_dir = ray_mat[:,i]
        ray_org = org_mat[:,i]
        t_hit, norm_vec = ray_line_obj_int(ray_dir, ray_org, Lens_obj_mat)
        if t_hit == False:
            continue
        #norm_vecs[0,i,:] = norm_vec
        def_ang_mat[0,i] = get_deflection_angle(ray_dir, norm_vec)
        n_lens = Lens_obj_mat[8,0]
        int_point = intercetion_point_fkt(ray_org, ray_dir, t_hit)
        int_mat[:,i,1] = int_point
        ray_dir = snells_law(ray_dir, norm_vec, 1, n_lens)
        # udregner ref fra ray_dir and norm_vec
        

        ray_org = int_point
        
        t_hit, norm_vec = ray_line_obj_int(ray_dir, ray_org, Lens_obj_mat)
        if t_hit == False:
            continue
        #[1,i,:] = norm_vec
        def_ang_mat[1,i] = get_deflection_angle(ray_dir, norm_vec)
        int_point = intercetion_point_fkt(ray_org, ray_dir, t_hit)
        int_mat[:,i,2] = int_point
        ray_dir = snells_law(ray_dir, norm_vec, n_lens, 1)
        
        t_plane = hit_plane_fkt(ray_org, ray_dir, np.array([plane_x,0]), np.array([1,0]))
        int_point = intercetion_point_fkt(ray_org, ray_dir, t_plane)
        int_mat[:,i,3] = int_point
    return int_mat, def_ang_mat

def plot_vectors_fkt(int_mat, Lines_obj_mat): #, norm_vecs):
    ax = plt.figure().add_subplot()
    for i in range(int(len(Lines_obj_mat[0,:]))):
        ax.plot(Lines_obj_mat[4:6,i], Lines_obj_mat[6:8,i], 'b')

    for i in range(len(int_mat[0,:,0])):
        ax.plot([int_mat[0,i,0], int_mat[0,i,1], int_mat[0,i,2], int_mat[0,i,3]] ,[int_mat[1,i,0], int_mat[1,i,1], int_mat[1,i,2], int_mat[1,i,3]], 'r')
    
    # for i in range(len(int_mat[0,:,0])):
    #     displacement1 = int_mat[:,i,1]
    #     displacement2 = int_mat[:,i,2]
    #     ax.plot([displacement1[0] - 0.5*norm_vecs[0,i,0], displacement1[0] + 0.5*norm_vecs[0,i,0]], [displacement1[1] - 0.5*norm_vecs[0,i,1], displacement1[1] + 0.5*norm_vecs[0,i,1]],'k')
    #     ax.plot([displacement2[0] - 0.5*norm_vecs[1,i,0], displacement2[0] + 0.5*norm_vecs[1,i,0]], [displacement2[1] - 0.5*norm_vecs[1,i,1], displacement2[1] + 0.5*norm_vecs[1,i,1]],'k')
        
    ax.set_xlim(-15, 30)
    ax.set_ylim(-6, 6)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()
    
    

def plot_lenses_fkt(int_mat1, int_mat2, Lines_obj_mat1, Lines_obj_mat2): #, norm_vecs):
    var = 3.2 
    ax = plt.figure().add_subplot()
    for i in range(int(len(Lines_obj_mat1[0,:]))):
        ax.plot(Lines_obj_mat1[4:6,i]+var, Lines_obj_mat1[6:8,i], 'b')
        
    for i in range(int(len(Lines_obj_mat2[0,:]))):
        ax.plot(Lines_obj_mat2[4:6,i]+var, Lines_obj_mat2[6:8,i], 'b')

    for i in range(len(int_mat1[0,:,0])):
        ax.plot([int_mat1[0,i,0]+var, int_mat1[0,i,1]+var, int_mat1[0,i,2]+var] ,[int_mat1[1,i,0], int_mat1[1,i,1], int_mat1[1,i,2]], 'r')
    
    for i in range(len(int_mat2[0,:,0])):
        ax.plot([int_mat2[0,i,0]+var, int_mat2[0,i,1]+var, int_mat2[0,i,2]+var, int_mat2[0,i,3]+var] ,[int_mat2[1,i,0], int_mat2[1,i,1], int_mat2[1,i,2], int_mat2[1,i,3]], 'r')
    
    # for i in range(len(int_mat[0,:,0])):
    #     displacement1 = int_mat[:,i,1]
    #     displacement2 = int_mat[:,i,2]
    #     ax.plot([displacement1[0] - 0.5*norm_vecs[0,i,0], displacement1[0] + 0.5*norm_vecs[0,i,0]], [displacement1[1] - 0.5*norm_vecs[0,i,1], displacement1[1] + 0.5*norm_vecs[0,i,1]],'k')
    #     ax.plot([displacement2[0] - 0.5*norm_vecs[1,i,0], displacement2[0] + 0.5*norm_vecs[1,i,0]], [displacement2[1] - 0.5*norm_vecs[1,i,1], displacement2[1] + 0.5*norm_vecs[1,i,1]],'k')
        
    ax.set_xlim(-12, 15)
    ax.set_ylim(-7.5, 7.5)
    ax.set_xlabel('X [cm]', fontsize = 16)
    ax.set_ylabel('Y [cm]',fontsize = 16)
    ax.set_aspect('equal')
    plt.show()