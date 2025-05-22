# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 09:13:41 2024

@author: thoms
"""

### compare rast with RT
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors 

I_rat = np.load('rast_sphere4_BPA_I.npy')
I_RT = np.load('sphere4_BPA_I.npy')

x_vec=np.linspace(0, 2, num=400) #Scene center er sat i (x=0,y=0)
y_vec=np.linspace(-1, 1, num=400)

I_diff = abs(I_rat) - abs(I_RT)

plt.pcolormesh(I_diff)
plt.axis('equal')
plt.show()

max_rast_I =  np.max(abs(I_rat))
max_RT_I =  np.max(abs(I_RT))
scale = max_rast_I / max_RT_I
I_diff_scale = abs(I_rat) - abs(I_RT)*scale

plt.pcolormesh(x_vec, y_vec, I_diff_scale)
plt.colorbar()
plt.axis('equal')
plt.show()

plt.pcolormesh(x_vec, y_vec, np.angle(I_rat))
plt.colorbar()
plt.axis('equal')
plt.show()

plt.pcolormesh(x_vec, y_vec, np.angle(I_RT))
plt.colorbar()
plt.axis('equal')
plt.show()

phase_dif = np.angle(I_rat) - np.angle(I_RT)
plt.pcolormesh(x_vec, y_vec, phase_dif)
plt.colorbar()
plt.axis('equal')
plt.show()

fig = plt.figure()
gs = fig.add_gridspec(1, 2, hspace=0.2, wspace = 0.4)
(ax1, ax2) = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Difference between \n Ray Tracing and Rasterization simulations', fontsize=16)
ax1.set_title('Intensity', fontsize=16)
ax2.set_title('Phase', fontsize=16)
#ax3.set_title('Difference')
plt1 = ax1.pcolormesh(x_vec, y_vec, I_diff_scale*1000, cmap='coolwarm')
plt2 = ax2.pcolormesh(x_vec, y_vec, phase_dif/np.pi, cmap='twilight_shifted')
#ax3.pcolormesh(x_vec, y_vec, I_diff_scale)
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax1.set_xticks([0,1,2])
ax1.set_yticks([-1,0,1])
ax2.set_xticks([0,1,2])
ax2.set_yticks([-1,0,1])
ax1.tick_params(labelsize=14)
ax2.tick_params(labelsize=14)
#ax3.set_aspect('equal')
ax1.set_xlabel('x [m]', fontsize=16)
ax1.set_ylabel('y [m]', fontsize=16)
ax2.set_xlabel('x [m]', fontsize=16)
cbar = plt.colorbar(plt1,ax=ax1, shrink=0.6)
cbar.set_label("Intensity (a.u.)",fontsize=14)
cbar2 = plt.colorbar(plt2,ax=ax2, shrink=0.6)
cbar2.set_label("Phase (rad)", fontsize=14)

#ax2.set_ylabel('y [m]', fontsize=16)
#ax1.label_outer


fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0.1, wspace = -0.5)
axs = gs.subplots(sharex=True, sharey=True)
fig.suptitle('Sharing both axes')

axs[0,0].pcolormesh(x_vec, y_vec, abs(I_RT)*scale)
axs[0,1].pcolormesh(x_vec, y_vec, abs(I_rat))
axs[1,0].pcolormesh(x_vec, y_vec, np.angle(I_RT))
axs[1,1].pcolormesh(x_vec, y_vec, np.angle(I_rat))
axs[0,0].set_aspect('equal', 'box')
axs[0,1].set_aspect('equal', 'box')
axs[1,0].set_aspect('equal', 'box')
axs[1,1].set_aspect('equal', 'box')
#pos = axs[0,0].imshow(axs[0,0], cmap='Blues')


custom_map_linlog = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#000000", "#3F3F3F","#737373","#989898","#B3B3B3","#C8C8C8","#D8D8D8","#E6E6E6","#F3F3F3","#FFFFFF"])


### GOOOD image!
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharey=True, sharex=True)
fig = plt.figure()
gs = fig.add_gridspec(2, 2, hspace=0.2, wspace = -0.5)
((ax1, ax2), (ax3, ax4)) = gs.subplots(sharex=True, sharey=True)
fig.suptitle('SAR images of a sphere', fontsize=16, y=1.05)
ax1.set_title('Ray Tracer', fontsize=16)
ax2.set_title('Rasterization', fontsize=16)


plt1 = ax1.pcolormesh(x_vec, y_vec, abs(I_RT)*scale*1000, cmap=custom_map_linlog)
plt2 = ax2.pcolormesh(x_vec, y_vec, abs(I_rat)*1000, cmap=custom_map_linlog)
plt3 = ax3.pcolormesh(x_vec, y_vec, np.angle(I_RT)/np.pi, cmap='twilight_shifted')
plt4 = ax4.pcolormesh(x_vec, y_vec, np.angle(I_rat)/np.pi, cmap='twilight_shifted')
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')
ax1.set_xticks([0,1,2])
ax1.tick_params(labelsize=14)
ax3.tick_params(labelsize=14)
ax4.tick_params(labelsize=14)
ax1.set_ylabel('y [m]', fontsize=16)
ax3.set_ylabel('y [m]', fontsize=16)
ax3.set_xlabel('x [m]', fontsize=16)
ax4.set_xlabel('x [m]', fontsize=16)
cbar = plt.colorbar(plt2,ax=ax2)
cbar.set_label("Intensity (a.u.)",fontsize=14)
cbar2 = plt.colorbar(plt4,ax=ax4)
cbar2.set_label("Phase (rad)", fontsize=14)










