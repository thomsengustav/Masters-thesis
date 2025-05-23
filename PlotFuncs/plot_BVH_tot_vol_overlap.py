# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:24:20 2025

@author: thoms
"""

'''
Plot Tot volume and overlap
'''

import numpy as np

t1 = 705
t2 = 307

vol1 = np.load('totVolVec_BVH.npy')
vol2 = np.load('totVolVec_BVHMopt.npy')

over1 = np.load('totOverVec_BVH.npy')
over2 = np.load('totOverVec_BVHMopt.npy')

layer = np.linspace(0,11, 12)

import matplotlib.pyplot as plt

plt.plot(layer, vol1, 'b')
plt.plot(layer, vol2, 'r')
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.suptitle('  ')
ax1.plot(layer, vol1)
ax1.plot(layer, vol2)
ax1.set_xlabel('BVH depth', fontsize = 14)
ax1.set_ylabel('Total BVH volume [$m^2$]', fontsize = 14)
ax1.set_xlim([0,11])
ax2.plot(layer, over1)
ax2.plot(layer, over2)
ax2.set_xlabel('BVH depth', fontsize = 14)
ax2.set_ylabel('Total BVH overlap volume [$m^2$]', fontsize = 14)
ax2.set_xlim([0, 11])
ax2.set_xticks(layer)
ax1.set_xticks(layer)
ax1.legend(['BVH', 'BVH w. mesh opt.'])
ax2.legend(['BVH', 'BVH w. mesh opt.'])
fig.tight_layout()
