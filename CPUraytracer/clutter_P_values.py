# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 10:32:36 2024

@author: thoms
"""

'''
library of clutter P-values
'''

import numpy as np

def get_P_vec_tree(pol_orientation):
    if pol_orientation == 'HH':
        #trees in X-band HH
        P_vec = np.array([-12.078, 1*10**(-6), -10, 4.574, 1.171, 0.583])
    elif pol_orientation == 'HV':
        #trees in X-band HV
        P_vec = np.array([88.003, -99.0, -0.05, 1,388, 6.204, -2.003])
    elif pol_orientation == 'VV':
        #trees in X-band VV
        P_vec = np.array([-11.751, 2*10**(-6), -10, 3.596, 2.033, 0.122])
    else:
        print('not valid polarization')
        return
    
    return P_vec