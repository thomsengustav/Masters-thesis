# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 11:39:31 2024

@author: thoms
"""

# intercetion with polygon (triangle)
# ray vector is given by r = P + t*b, V1, V2, V3 are the vetecies of the triangle
# tm maximum allowed values of t inside the bounding volume.
def func_triangle_intersection_t(P, b, V1, V2, V3, tm):
    # calculate matrix elements
    M11, M21, M31 = V1[0] - V2[0], V1[1] - V2[1], V1[2] - V2[2]
    M12, M22, M32 = V1[0] - V3[0], V1[1] - V3[1], V1[2] - V3[2]
    M13, M23, M33 = b[0], b[1], b[2]
    
    E1, E2, E3 = V1[0] - P[0],  V1[1] - P[1],  V1[2] - P[2]
    
    TM1, TM2, TM3 = M22*M33 - M23*M32, M13*M32 - M12*M33, M12*M23 - M22*M13
    Tt1, Tt2, Tt3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    M_denominator = M11*TM1 + M21*TM2 + M31*TM3
    # calculate t
    t_tri = -(M32*Tt1 + M22*Tt2 + M12*Tt3)/M_denominator
    # check if  0 < t < tm
    if t_tri < 0.00001:
        return 0
    if t_tri > tm:
        return 0
    
    beta = (E1*TM1 + E2*TM2 + E3*TM3)/M_denominator
    
    #check if 0 < beta < 1
    if beta < 0:
        return 0
    if beta > 1:
        return 0
    
    TG1, TG2, TG3 = M11*E2 - E1*M21, E1*M31 - M11*E3, M21*E3 - E2*M31
    gamma = (M33*TG1 + M23*TG2 + M13*TG3)/M_denominator
    
    # check if gamma > 0 and gamma + beta < 1
    if gamma < 0:
        return 0
    if gamma > 1 - beta:
        return 0
    
    return t_tri



