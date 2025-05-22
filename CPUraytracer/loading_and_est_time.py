# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 12:48:38 2025

@author: thoms
"""

'''
estimate time and loading bar
'''

import time

def loading_bar(sar_counter, azimuth_steps):
    
    precentage = int(sar_counter / azimuth_steps * 100 ) 
    string = 'Calculating SAR: '
    procent_plot_var = int(precentage / 10)
    bar = ''
    for i in range(10):
        value = procent_plot_var - i
        if value > 0:
            bar = bar + '■'
        else:
            bar = bar + '□'
    full_bar = string + bar + ' ' + str(precentage) + '%'
    print(full_bar)

def get_time_est(time_1, tot_time, azimuth_steps, sar_counter):
    time_2 = time.time()
    time_loop = time_2 - time_1
    tot_time = time_loop + tot_time
    multiplier =  azimuth_steps / sar_counter 
    decimal_done = 1 / multiplier
    est_time = tot_time * multiplier * (1 - decimal_done)
    string = 'Estimated time remaning: '
    est_time_hours = int(est_time / 3600)
    time_minus_h = est_time % 3600
    est_time_min = int(time_minus_h / 60)
    time_minus_m = time_minus_h % 60
    est_time_sec = int(time_minus_m)
    
    fin_string = string + str(est_time_hours) + ' hours, ' + str(est_time_min) + ' min, ' + str(est_time_sec) + ' sec'
    print(fin_string)
    return tot_time