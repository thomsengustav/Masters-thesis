# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:07:58 2025

@author: thoms
"""

'''
calculate BVH total volume and overlap functions
'''

import numpy as np
from numba import njit

# compute number of layers of BVH
@njit
def getNumLayers(box_node_mat):
    n = len(box_node_mat[0,:])
    log2 = np.log(2)
    logn1 = np.log(n+1)
    return int(logn1 / log2 - 1)

# cumpute total BVH volume for each layer
@njit
def getTotVolume(box_node_mat):
    nLayers = getNumLayers(box_node_mat) + 1
    totVolVec = np.zeros(nLayers)
    
    matStart = 0
    for i in range(nLayers):
        nBoxes = int(2**(i))
        tot = 0
        for n in range(nBoxes):
            m = n + matStart
            xlen = abs(box_node_mat[0,m] - box_node_mat[1,m])
            ylen = abs(box_node_mat[2,m] - box_node_mat[3,m])
            zlen = abs(box_node_mat[4,m] - box_node_mat[5,m])
            tot += xlen * ylen * zlen
        totVolVec[i] = tot
        matStart = nBoxes + matStart
    return totVolVec

# compute total overlav volume for each layer
@njit
def getTotOverlap(box_node_mat, totVolVec):
    nLayers = getNumLayers(box_node_mat) + 1
    totOverlapVec = np.zeros(nLayers)
    
    matStart = 0
    for i in range(nLayers):
        nBoxes = int(2**(i))
        for n in range(nBoxes):
            m = n + matStart
            
            xMinVec = np.zeros(2)
            xMaxVec = np.zeros(2)
            yMinVec = np.zeros(2)
            yMaxVec = np.zeros(2)
            zMinVec = np.zeros(2)
            zMaxVec = np.zeros(2)
            
            xMinVec[0] = np.min(np.array([box_node_mat[0,m], box_node_mat[1,m]]))
            xMaxVec[0] = np.max(np.array([box_node_mat[0,m], box_node_mat[1,m]]))
            yMinVec[0] = np.min(np.array([box_node_mat[2,m], box_node_mat[3,m]]))
            yMaxVec[0] = np.max(np.array([box_node_mat[2,m], box_node_mat[3,m]]))
            zMinVec[0] = np.min(np.array([box_node_mat[4,m], box_node_mat[5,m]]))
            zMaxVec[0] = np.max(np.array([box_node_mat[4,m], box_node_mat[5,m]]))
            
            for k in range(nBoxes):
                l = k + matStart
                
                xMinVec[1] = np.min(np.array([box_node_mat[0,l], box_node_mat[1,l]]))
                xMaxVec[1] = np.max(np.array([box_node_mat[0,l], box_node_mat[1,l]]))
                yMinVec[1] = np.min(np.array([box_node_mat[2,l], box_node_mat[3,l]]))
                yMaxVec[1] = np.max(np.array([box_node_mat[2,l], box_node_mat[3,l]]))
                zMinVec[1] = np.min(np.array([box_node_mat[4,l], box_node_mat[5,l]]))
                zMaxVec[1] = np.max(np.array([box_node_mat[4,l], box_node_mat[5,l]]))
                
                xMinOverlap = np.max(xMinVec)
                xMaxOverlap = np.min(xMaxVec)
                yMinOverlap = np.max(yMinVec)
                yMaxOverlap = np.min(yMaxVec)
                zMinOverlap = np.max(zMinVec)
                zMaxOverlap = np.min(zMaxVec)
                
                xlen = xMaxOverlap - xMinOverlap
                if xlen < 0:
                    continue
                ylen = yMaxOverlap - yMinOverlap
                if ylen < 0:
                    continue
                zlen = zMaxOverlap - zMinOverlap
                if zlen < 0:
                    continue

                totOverlapVec[i] += xlen * ylen * zlen
        matStart = nBoxes + matStart
    
    totOverlapVec = totOverlapVec - totVolVec

    return totOverlapVec / 2


