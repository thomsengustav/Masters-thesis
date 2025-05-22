# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 13:04:05 2024

@author: thoms
"""

### fresnel test


import numpy as np
import matplotlib.pyplot as plt

ang_ls = np.linspace(0, 90, 90*4)

def T_s_fkt(n0, n1, ang):
    ang_rad = ang * np.pi / 180
    ang_t = np.arcsin(n0/n1*np.sin(ang_rad))
    t_s = (2*n0*np.cos(ang_rad)) / (n0*np.cos(ang_rad) + n1*np.cos(ang_t))
    T = t_s**2 * (n1 * np.cos(ang_t)) / (n0 * np.cos(ang_rad))
    return t_s

def R_s_fkt(n0, n1, ang):
    ang_rad = ang * np.pi / 180
    ang_t = np.arcsin(n0/n1*np.sin(ang_rad))
    r_s = (n0*np.cos(ang_rad) - n1*np.cos(ang_t)) / (n0*np.cos(ang_rad) + n1*np.cos(ang_t))
    R = r_s**2
    return r_s

# R_ls = R_s_fkt(1, 1.5, ang_ls)
# T_ls = T_s_fkt(1, 1.5, ang_ls)

# plt.plot(ang_ls, R_ls)
# plt.plot(ang_ls, T_ls)
# plt.show()

def T_p_fkt(n0, n1, ang):
    ang_rad = ang * np.pi / 180
    ang_t = np.arcsin(n0/n1*np.sin(ang_rad))
    t_s = (2*n0*np.cos(ang_rad)) / (n1*np.cos(ang_rad) + n0*np.cos(ang_t))
    T = t_s**2 * (n1 * np.cos(ang_t)) / (n0 * np.cos(ang_rad))
    return t_s

def R_p_fkt(n0, n1, ang):
    ang_rad = ang * np.pi / 180
    ang_t = np.arcsin(n0/n1*np.sin(ang_rad))
    r_s = (n1*np.cos(ang_rad) - n0*np.cos(ang_t)) / (n1*np.cos(ang_rad) + n0*np.cos(ang_t))
    R = r_s**2
    return r_s

n1 = 600 +1j*600

#n1 = 10 +1j*10
#n1=1.5
R_ls = R_p_fkt(1, n1, ang_ls)
T_ls = T_p_fkt(1, n1, ang_ls)

R_ls2 = R_s_fkt(1, n1, ang_ls)
T_ls2 = T_s_fkt(1, n1, ang_ls)


plt.plot(ang_ls, R_ls)
plt.plot(ang_ls, T_ls)
plt.plot(ang_ls, R_ls2)
plt.plot(ang_ls, T_ls2)
plt.show()

fig, ax = plt.subplots(figsize=(5,4))
ax.set_title('Air to metal, $n_M=300+i300$', fontsize=16)
plt1, =ax.plot(ang_ls, R_ls, label = '$r_p$', color='b', linewidth=2.5)
plt2, =ax.plot(ang_ls, T_ls, label = '$t_p$', color='b', linestyle='dashed', linewidth=2.5)
plt3, =ax.plot(ang_ls, R_ls2, label = '$r_s$', color='r', linewidth=2.5)
plt4, =ax.plot(ang_ls, T_ls2, label = '$t_s$', color='r', linestyle='dashed',linewidth=2.5 )
ax.legend(handles=[plt1, plt2, plt3, plt4], fontsize=14, loc=3)
ax.set_ylim([-1,1])
ax.set_xlim([0,90])
ax.set_xlabel("Angle of incidence [deg]", fontsize=16)







