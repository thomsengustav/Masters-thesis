# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 11:52:58 2024

@author: thoms
"""

#### ray scattering on surfaces test!
'''
Script for plotting different scattering profiles
'''


import numpy as np
from numba import njit, prange

def diffuse_spread_fkt(normal, N):
    x, y = np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    
    vector_length_inverse = 1 / np.sqrt(x**2 + y**2)
    
    diffuse_mat = np.zeros((N,2))
    diffuse_mat[:,0] = np.multiply(x, vector_length_inverse) + normal[0]
    diffuse_mat[:,1] = np.multiply(y, vector_length_inverse) + normal[1]
    
    norm_diff = 1 / np.sqrt(diffuse_mat[:,0]**2 + diffuse_mat[:,1]**2)
    diffuse_mat[:,0] = np.multiply(diffuse_mat[:,0], norm_diff)
    diffuse_mat[:,1] = np.multiply(diffuse_mat[:,1], norm_diff)
    return diffuse_mat

def ref_2d(incoming_ang): # normal along y-axis
    inc_vec = - np.array([np.cos(incoming_ang/180*np.pi), np.sin(incoming_ang/180*np.pi)])
    ray_surface_dot = np.dot(inc_vec, np.array([0,1]))
    ref_direction = inc_vec - 2 * ray_surface_dot * np.array([0,1])
    return inc_vec, ref_direction


def norm_ref_simple(N, spread):
    x, y = np.random.normal(0, spread, N), np.random.normal(0, spread, N)
    
    x_dis = x
    y_dis = y + 1
    vector_length_inverse = 1 / np.sqrt(x_dis**2 + y_dis**2)
    diffuse_mat = np.zeros((N,2))
    diffuse_mat[:,0] = np.multiply(x_dis, vector_length_inverse)
    diffuse_mat[:,1] = np.multiply(y_dis, vector_length_inverse)
    return diffuse_mat

def ref_diff_alpha(ref_vec, normal, N, alpha):
    x, y = np.random.normal(0, 1, N), np.random.normal(0, 1, N)
    
    vector_length_inverse = 1 / np.sqrt(x**2 + y**2)
    
    dif_x = np.multiply(x, vector_length_inverse) + normal[0]
    dif_y = np.multiply(y, vector_length_inverse) + normal[1]
    
    norm_diff = 1 / np.sqrt(dif_x**2 + dif_y**2)
    norm_dif_x = np.multiply(dif_x, norm_diff)
    norm_dif_y = np.multiply(dif_y, norm_diff)
    
    scat_mat = np.seros((N,2))
    scat_mat[:,0] = alpha * norm_dif_x + (1 - alpha) * ref_vec[0]
    scat_mat[:,1] = alpha * norm_dif_y + (1 - alpha) * ref_vec[1]
    return scat_mat
    
    
def D2_plot_vector_opacity_fkt(x, y, vector_origin, xlim, ylim):
    ax = plt.figure().add_subplot()

    for i in range(len(x)):
        ax.plot([vector_origin[0], x[i]] ,[vector_origin[1], y[i]],'r',alpha=0.01)
        
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()


def ray_pr_ang(ray_matrix):
    dit_variance = np.cos(0.5/180*np.pi) # 1 degree
    angle_interval = np.linspace(0, 180, 100)
    dit_variance = np.cos((angle_interval[1]-angle_interval[0])/2/180*np.pi) # 1 degree
    ang_counter = np.zeros(len(angle_interval))
    
    for i in range(len(angle_interval)):
        ang_pointer_vec = np.array([np.cos(angle_interval[i]/180*np.pi), np.sin(angle_interval[i]/180*np.pi)])
        
        dot_x = ang_pointer_vec[0] * ray_matrix[:,0]
        dot_y = ang_pointer_vec[1] * ray_matrix[:,1]
        dot_res = dot_x + dot_y
        
        ang_index = np.where(dot_res >= dit_variance)
        ang_counter[i] = len(ang_index[0])
    
    return angle_interval, ang_counter

from plot_vector_fkt import D2_plot_vector_fkt
import matplotlib.pyplot as plt
normal = np.array([0,1])
mat = diffuse_spread_fkt(normal, 100000)
mat = norm_ref_simple(50000, 0.3)
ang_vec, len_vec = ray_pr_ang(mat)

ang_pointer_vec = np.array([np.cos(0/180*np.pi), np.sin(0/180*np.pi)])
ray_matrix = mat
dot_x = ang_pointer_vec[0] * ray_matrix[:,0]
dot_y = ang_pointer_vec[1] * ray_matrix[:,1]
dot_res = dot_x + dot_y
dit_variance = np.cos(1/180*np.pi) # 1 degree
ang_index = np.where(dot_res >= dit_variance)
print(ang_index[0])

# index = ang_index[0][0]
# plt.plot([ang_pointer_vec[0], 0], [ang_pointer_vec[1], 0])
# plt.plot([ray_matrix[index,0],0],[ray_matrix[index,1],0])
# plt.show()


def circular_ray_plot(angle_interval, ang_counter):
    ax = plt.figure().add_subplot()
    
    x_vec = np.zeros(len(angle_interval)+2)
    y_vec = np.zeros(len(angle_interval)+2)
    x_vec[0], x_vec[-1] = 0, 0
    y_vec[0], y_vec[-1] = 0, 0
    max_len = max(ang_counter)
    for i in range(len(angle_interval)):
        ang_pointer_vec = np.array([np.cos(angle_interval[i]/180*np.pi), np.sin(angle_interval[i]/180*np.pi)])
        norm_point = ang_pointer_vec * ang_counter[i] / max_len
        x_vec[i+1] = norm_point[0]
        y_vec[i+1] = norm_point[1]
        
    ax.plot(x_vec, y_vec)
        
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(0, 1.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')

    plt.show()


circular_ray_plot(ang_vec, len_vec)


def fibonacci_sphere(samples=1000):

    points = np.zeros((samples,3))
    phi = np.pi * (np.sqrt(5.) - 1.)  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points[i,0] = x
        points[i,1] = y
        points[i,2] = z

    return points

points = fibonacci_sphere(samples=12000)

def get_halfsphere(points):
    N = int(len(points[:,0]))
    New_points = np.zeros((N,3))
    counter  = 0
    for i in range(N):
        z = points[i,2]
        if z < 0:
            continue
        else:
            New_points[counter,:] = points[i,:]
            counter += 1
    points = New_points[:counter,:]
    return points 

New_points = get_halfsphere(points)

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(points[:,0], points[:,1], points[:,2],)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(New_points[:,0], New_points[:,1], New_points[:,2],)
# plt.show()



#alpha controls spreading parm, N number of rays, theta angle of incidence
theta = 30
N = 1000
alpha = 0.8

theta_rad = theta * np.pi / 180



def plot_vectors(scat_vec, elevation_angle, azimuth_angle):
    N = int(len(scat_vec[:,0]))
    
    N = 10
    for i in range(N):
        ax = plt.figure().add_subplot(projection='3d')
        x = scat_vec[i,0]
        y = scat_vec[i,1]
        z = scat_vec[i,2]
        lent = np.sqrt(x**2+y**2+z**2)
        ax.plot([0, x] ,[0, y], [0, z])
        
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_zlim(-0.5,2.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
    
        ax.view_init(elev=elevation_angle, azim=azimuth_angle, roll=0)
        plt.show()

# alpha method
def get_scat_vectors(alpha, N, theta):
    theta_rad = theta * np.pi / 180
    z = np.sin(theta_rad)
    x = -np.cos(theta_rad)
    in_vec = -np.array([x,0,z]) # located in zx-plane
    n_vec = np.array([0,0,1])
    ref_vec = in_vec - 2*(np.dot(in_vec,n_vec)*n_vec)

    ref_vec_mat = np.zeros((N,3))
    ref_vec_mat[0:-1,:] = ref_vec * (1-alpha)

    dif_x, dif_y, dif_z = np.random.normal(0,1,N), np.random.normal(0,1,N), np.random.normal(0,1,N)

    Len_inv_dif = 1 / np.sqrt(dif_x**2+dif_y**2+dif_z**2)

    r_x, r_y, r_z = dif_x*Len_inv_dif, dif_y*Len_inv_dif, dif_z*Len_inv_dif + 1   

    Len_inv_vec = 1 / np.sqrt(r_x**2+r_y**2+r_z**2)

    dif_vec = np.zeros((N,3))
    dif_vec[:,0] = r_x * Len_inv_vec * alpha
    dif_vec[:,1] = r_y * Len_inv_vec * alpha
    dif_vec[:,2] = r_z * Len_inv_vec * alpha


    scat_vec = dif_vec + ref_vec_mat

    Len_inv_scat = 1 / np.sqrt(scat_vec[:,0]**2+scat_vec[:,1]**2+scat_vec[:,2]**2)

    scat_vec[:,0] = scat_vec[:,0] * Len_inv_scat 
    scat_vec[:,1] = scat_vec[:,1] * Len_inv_scat 
    scat_vec[:,2] = scat_vec[:,2] * Len_inv_scat 
    
    return scat_vec

# gamma method
# backscattering profile with same alpha 
def get_scat_vectors1(alpha, beta, N, theta):
    theta_rad = theta * np.pi / 180
    z = np.sin(theta_rad)
    x = -np.cos(theta_rad)
    in_vec = -np.array([x,0,z]) # located in zx-plane
    n_vec = np.array([0,0,1])
    ref_vec = in_vec - 2*(np.dot(in_vec,n_vec)*n_vec)
    
    
    N_back = int(beta*N)
    N_out = N-N_back
    
    
    ref_vec_mat_back = np.zeros((N_back,3))
    ref_vec_mat_out = np.zeros((N_out,3))
    ref_vec_mat_back[0:-1,:] = -in_vec * (1-alpha)
    ref_vec_mat_out[0:-1,:] = ref_vec * (1-alpha)
    

    dif_x, dif_y, dif_z = np.random.normal(0,1,N), np.random.normal(0,1,N), np.random.normal(0,1,N)

    Len_inv_dif = 1 / np.sqrt(dif_x**2+dif_y**2+dif_z**2)
    
    r_x, r_y, r_z = dif_x*Len_inv_dif, dif_y*Len_inv_dif, dif_z*Len_inv_dif + 1   
    
    Len_inv_vec = 1 / np.sqrt(r_x**2+r_y**2+r_z**2)
    
    dif_vec = np.zeros((N,3))
    dif_vec[:,0] = r_x * Len_inv_vec * alpha
    dif_vec[:,1] = r_y * Len_inv_vec * alpha
    dif_vec[:,2] = r_z * Len_inv_vec * alpha

    
    scat_vec_back = dif_vec[0:N_back,:] + ref_vec_mat_back
    scat_vec_out = dif_vec[N_back:,:] + ref_vec_mat_out
    
    Len_inv_scat_back = 1 / np.sqrt(scat_vec_back[:,0]**2+scat_vec_back[:,1]**2+scat_vec_back[:,2]**2)
    Len_inv_scat_out = 1 / np.sqrt(scat_vec_out[:,0]**2+scat_vec_out[:,1]**2+scat_vec_out[:,2]**2)

    scat_vec = np.zeros((N,3))
    scat_vec[0:N_back,0] = scat_vec_back[:,0] * Len_inv_scat_back 
    scat_vec[0:N_back,1] = scat_vec_back[:,1] * Len_inv_scat_back 
    scat_vec[0:N_back,2] = scat_vec_back[:,2] * Len_inv_scat_back 
    
    scat_vec[N_back:,0] = scat_vec_out[:,0] * Len_inv_scat_out 
    scat_vec[N_back:,1] = scat_vec_out[:,1] * Len_inv_scat_out 
    scat_vec[N_back:,2] = scat_vec_out[:,2] * Len_inv_scat_out 
    
    return scat_vec

# lambda method
def get_scat_vectors2(alpha, gamma, N, theta):
    theta_rad = theta * np.pi / 180
    z = np.sin(theta_rad)
    x = -np.cos(theta_rad)
    in_vec = -np.array([x,0,z]) # located in zx-plane
    n_vec = np.array([0,0,1])
    ref_vec = in_vec - 2*(np.dot(in_vec,n_vec)*n_vec)
    
    
    N_back = int(gamma*N)
    N_out = N-N_back
    
    
    ref_vec_mat_back = np.zeros((N_back,3))
    ref_vec_mat_out = np.zeros((N_out,3))
    ref_vec_mat_back[0:-1,:] = np.array([0,0,0])
    ref_vec_mat_out[0:-1,:] = ref_vec * (1-alpha)
    

    dif_x, dif_y, dif_z = np.random.normal(0,1,N), np.random.normal(0,1,N), np.random.normal(0,1,N)

    Len_inv_dif = 1 / np.sqrt(dif_x**2+dif_y**2+dif_z**2)
    
    r_x, r_y, r_z = dif_x*Len_inv_dif, dif_y*Len_inv_dif, dif_z*Len_inv_dif + 1   
    
    Len_inv_vec = 1 / np.sqrt(r_x**2+r_y**2+r_z**2)
    
    dif_vec = np.zeros((N,3))
    dif_vec[:,0] = r_x * Len_inv_vec * alpha
    dif_vec[:,1] = r_y * Len_inv_vec * alpha
    dif_vec[:,2] = r_z * Len_inv_vec * alpha

    
    scat_vec_back = dif_vec[0:N_back,:] + ref_vec_mat_back
    scat_vec_out = dif_vec[N_back:,:] + ref_vec_mat_out
    
    Len_inv_scat_back = 1 / np.sqrt(scat_vec_back[:,0]**2+scat_vec_back[:,1]**2+scat_vec_back[:,2]**2)
    Len_inv_scat_out = 1 / np.sqrt(scat_vec_out[:,0]**2+scat_vec_out[:,1]**2+scat_vec_out[:,2]**2)

    scat_vec = np.zeros((N,3))
    scat_vec[0:N_back,0] = scat_vec_back[:,0] * Len_inv_scat_back 
    scat_vec[0:N_back,1] = scat_vec_back[:,1] * Len_inv_scat_back 
    scat_vec[0:N_back,2] = scat_vec_back[:,2] * Len_inv_scat_back 
    
    scat_vec[N_back:,0] = scat_vec_out[:,0] * Len_inv_scat_out 
    scat_vec[N_back:,1] = scat_vec_out[:,1] * Len_inv_scat_out 
    scat_vec[N_back:,2] = scat_vec_out[:,2] * Len_inv_scat_out 
    
    return scat_vec

print('nu regner jeg scat_vec')
scat_vec = get_scat_vectors2(0.7, 1,6000000, theta)
    

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(New_points[:,0], New_points[:,1], New_points[:,2],)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(scat_vec[:,0], scat_vec[:,1], scat_vec[:,2],)
# plt.show()



@njit(parallel=True)
def count_points(scat_vec, sphere_points):
    N_vecs = int(len(scat_vec[:,0]))
    N_p = int(len(sphere_points[:,0]))
    index_vec = np.zeros(N_p)
    
    for n in prange(N_vecs):
        V = scat_vec[n,:]
        dist_p = 1
        for i in range(N_p):
            p = sphere_points[i,:]
            dif_pV = V-p
            dist = np.sqrt(np.dot(dif_pV,dif_pV))
            if dist < dist_p:
                dist_p = dist
                index = i
        index_vec[index] += 1
    return index_vec

print('nu regner jeg index_vec')
index_vec = count_points(scat_vec, New_points)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = New_points[:,0]
y = New_points[:,1]
z = New_points[:,2]
c = index_vec

img = ax.scatter(x, y, z, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()


# # Scale
max_index = np.max(index_vec)

x_scale = x *  index_vec / max_index
y_scale = y *  index_vec / max_index
z_scale = z *  index_vec / max_index


# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x_scale, y_scale, z_scale,)
# plt.xlim([-1,1])
# plt.ylim([-1,1])
# plt.zlim([0,2])
# plt.show()

M = int(len(index_vec))
data_mat = np.zeros((4,M))
data_mat[0,:] = x_scale
data_mat[1,:] = y_scale
data_mat[2,:] = z_scale
data_mat[3,:] = index_vec

data_mat_uvec = np.zeros((5, M))
Len_inv = 1 / np.sqrt(data_mat[0,:]**2+data_mat[1,:]**2+data_mat[2,:]**2) 
data_mat_uvec[0,:] = x_scale * Len_inv
data_mat_uvec[1,:] = y_scale * Len_inv
data_mat_uvec[2,:] = z_scale * Len_inv 
data_mat_uvec[3,:] = index_vec
data_mat_uvec[4,:] = Len_inv

test1 = data_mat_uvec[1,:]
test = data_mat_uvec[1,:].argsort()

data_new = data_mat[:,np.flip(data_mat_uvec[2,:].argsort())]
# data_new[0,:] = data_new[0,:] / data_new[4,:]
# data_new[1,:] = data_new[1,:] / data_new[4,:]
# data_new[2,:] = data_new[2,:] / data_new[4,:]
data_flip = np.flip(data_new,1)

x_flip = data_flip[0,:]
y_flip = data_flip[1,:]
z_flip = data_flip[2,:]
index_flip = data_flip[3,:]

blue = [73, 27, 191]
purple = [166, 17, 168]
red = [191, 19, 39]
orange = [207, 132, 27]
yellow = [247, 247, 25] 

r = np.array([73, 166, 191, 207, 247])
g = np.array([27, 17, 19, 132, 247])
b = np.array([191, 168, 39, 27, 25])

val = np.array([0,1/4,2/4,3/4,4/4])

# plt.plot(val,r, 'r')
# plt.plot(val,g, 'g')
# plt.plot(val,b, 'b')
# plt.show()

r_int = np.interp(0.3, val, r)

print('nu plotter jeg')


ax = plt.figure(figsize=(8,8)).add_subplot(projection='3d')
N = int(len(x_scale))
z = np.sin(theta_rad) *0.85
x = -np.cos(theta_rad) *0.85
ax.plot([0, x] ,[0, 0], [0, z], 'k', linewidth='4')
for i in range(N):
    x = x_flip[i]
    y = y_flip[i]
    z = z_flip[i]
    index = index_flip[i] / max_index
    r_int = np.interp(index, val, r)
    g_int = np.interp(index, val, g)
    b_int = np.interp(index, val, b)
    ax.plot([0, x] ,[0, y], [0, z], linewidth='3', color=(r_int/255, g_int/255, b_int/255)) #, linewidth='3')

ax.set_xlim(-0.8, 0.8)
ax.set_ylim(-0.8, 0.8)
ax.set_zlim(0,1)
ax.axes.tick_params(labelleft = False, 
                labelbottom = False)
ax.set_aspect('equal') # , adjustable='box')
ax.set_xlabel('X', fontsize=16)
ax.set_ylabel('Y', fontsize=16)
ax.set_zlabel('Z', fontsize=16)
#ax.set_title('alpha = 0.8')

ax.view_init(elev=90, azim=90, roll=0)
plt.show()
