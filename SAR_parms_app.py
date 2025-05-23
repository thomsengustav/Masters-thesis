# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 12:46:05 2025

@author: thoms
"""

import tkinter as tk
from decimal import Decimal
import numpy as np

def get_dr(B, c):
    return c / (2*B)

def get_dx(theta_int, waveL):
    theta_int_rad = theta_int / 180 * np.pi
    return waveL / (2*theta_int_rad)

def get_rande_D(waveL, delta_azi):
    delta_azi_rad = delta_azi / 180 * np.pi
    return waveL / (2*delta_azi_rad)

def get_SAR_constraints(f, B, theta_int, N_int):
    SAR_parm_vec = np.zeros(9)
    c = 299792458 # m/s
    
    waveL = c / f
    delta_azi = theta_int / N_int
    
    SAR_parm_vec[0] = f
    SAR_parm_vec[1] = B
    SAR_parm_vec[2] = waveL
    SAR_parm_vec[3] = theta_int
    SAR_parm_vec[4] = N_int
    SAR_parm_vec[5] = delta_azi
    SAR_parm_vec[6] = get_dr(B, c)
    SAR_parm_vec[7] = get_dx(theta_int, waveL)
    SAR_parm_vec[8] = get_rande_D(waveL, delta_azi)
    
    return SAR_parm_vec

f = 0
B = 0
theta = 0
N_A = 0
f_units = 0 # 0 = THz, 1 = GHz
waveL = 0
delta_azi = 0
dr = 0
dx = 0
D = 0


def get_f(f, f_units):
    if f_units == 0:
        return f*1e12
    elif f_units == 1:
        return f*1e9

def set_funits_THz():
    global f_units
    f_units = 0

def set_funits_GHz():
    global f_units
    f_units=1

def print_funit():
    global f_units
    print(f_units)

def set_f():
    global f
    global f_units
    
    text = entryf.get()
    if text:
        f = float(text)
        f = get_f(f, f_units)
        f_label = '%.3E' % Decimal(str(f))
        entryf.delete(0,tk.END)
        lblfVal.config(text =f_label)
        
def set_B():
    global B
    global f_units
    text = entryB.get()
    if text:
        B = float(text)
        B = get_f(B, f_units)
        B_label = '%.3E' % Decimal(str(B))
        entryB.delete(0,tk.END)
        lblBVal.config(text =B_label)
        
def set_theta():
    global theta
    text = entryTheta.get()
    if text:
        theta = float(text)
        entryTheta.delete(0,tk.END)
        lblThetaVal.config(text = str(theta))
        
def set_NA():
    global N_A
    text = entryNA.get()
    if text:
        N_A = float(text)
        entryNA.delete(0,tk.END)
        lblNAVal.config(text= str(N_A))
    
def button_parms():
    global f
    global B
    global theta
    global N_A
    
    parms_vec = get_SAR_constraints(f, B, theta, N_A)
    
    f_label = '%.3E' % Decimal(str(f))
    lblfSARval.config(text = f_label)
    
    B_label = '%.3E' % Decimal(str(B))
    lblBSARval.config(text = B_label)
    
    waveL_label = '%.3E' % Decimal(str(parms_vec[2]))
    lblwaveLSARval.config(text = waveL_label)
    
    lblThetaSARval.config(text = str(theta))
    
    lblNASARval.config(text = str(N_A))
    
    dA_lavel = '%.3E' % Decimal(str(parms_vec[5]))
    lbldASARval.config(text = dA_lavel)
    dr_lbl = '%.3E' % Decimal(str(parms_vec[6]))
    lbldrSARval.config(text = dr_lbl)
    dx_lbl = '%.3E' % Decimal(str(parms_vec[7]))
    lbldxSARval.config(text = dx_lbl)
    D_lbl = '%.3E' % Decimal(str(parms_vec[8]))
    lblDSARval.config(text = D_lbl)
    
    
    
    
    

root = tk.Tk()
root.title("SAR parameters")

frame1 = tk.Frame(root)
frame1.grid(row=0,column=0)


lblfUnits = tk.Label(frame1, text = 'Pick frequency units')
lblfUnits.grid(row=0, column=0)

btnfUnitsTHz = tk.Button(frame1, text='[THz]', command = set_funits_THz)
btnfUnitsTHz.grid(row=0,column=1)
btnfUnitsGHz = tk.Button(frame1, text='[GHz]', command = set_funits_GHz)
btnfUnitsGHz.grid(row=0,column=2)

lblf = tk.Label(frame1, text = 'Pick frequency units')
lblfUnits.grid(row=0, column=0)

# btn_print=tk.Button(frame1, text = 'print', command = print_funit)
# btn_print.grid(row=1, column = 0)

lblf = tk.Label(frame1, text = 'Set frequency')
lblf.grid(row=1,column=0)
entryf = tk.Entry(frame1)
entryf.grid(row=1, column=1)
btnf = tk.Button(frame1, text='save', command = set_f)
btnf.grid(row=1,column=2,  sticky='ew')

lblB = tk.Label(frame1, text = 'Set bandwith')
lblB.grid(row=2,column=0)
entryB = tk.Entry(frame1)
entryB.grid(row=2, column=1)
btnB = tk.Button(frame1, text='save', command = set_B)
btnB.grid(row=2,column=2,  sticky='ew')

lblTheta = tk.Label(frame1, text = 'Set integration angle [deg]')
lblTheta.grid(row=3,column=0)
entryTheta = tk.Entry(frame1)
entryTheta.grid(row=3, column=1)
btnTheta = tk.Button(frame1, text='save', command = set_theta)
btnTheta.grid(row=3,column=2,  sticky='ew')

lblNA = tk.Label(frame1, text = 'Set azimuth steps')
lblNA.grid(row=4,column=0)
entryNA = tk.Entry(frame1)
entryNA.grid(row=4, column=1)
btnNA = tk.Button(frame1, text='save', command = set_NA)
btnNA.grid(row=4,column=2,  sticky='ew')

lblSpace = tk.Label(frame1, text ='        ')
lblSpace.grid(row=0,column=4)

lblSpace = tk.Label(frame1, text ='              ')
lblSpace.grid(row=0,column=6)

lblCurrentVal = tk.Label(frame1, text ='Current values')
lblCurrentVal.grid(row=0,column=5)

lblfVal = tk.Label(frame1, text =str(f))
lblfVal.grid(row=1,column=5)

lblBVal = tk.Label(frame1, text =str(B))
lblBVal.grid(row=2,column=5)

lblThetaVal = tk.Label(frame1, text =str(theta))
lblThetaVal.grid(row=3,column=5)

lblNAVal = tk.Label(frame1, text =str(N_A))
lblNAVal.grid(row=4,column=5)

btn_parms = tk.Button(frame1, text='Get SAR parameters', command = button_parms)
btn_parms.grid(row=5, column=1, columnspan = 2, sticky='ew')

frame2 = tk.Frame(root)
frame2.grid(row=1,column=0)

lblfSAR = tk.Label(frame2, text ='f [Hz]')
lblfSAR.grid(row=0,column=0)
lblfSARval = tk.Label(frame2, text =str(f))
lblfSARval.grid(row=1,column=0)

lblBSAR = tk.Label(frame2, text ='B [Hz]')
lblBSAR.grid(row=0,column=1)
lblBSARval = tk.Label(frame2, text =str(f))
lblBSARval.grid(row=1,column=1)

lblwaveLSAR = tk.Label(frame2, text ='λ [m]')
lblwaveLSAR.grid(row=0,column=2)
lblwaveLSARval = tk.Label(frame2, text =str(waveL))
lblwaveLSARval.grid(row=1,column=2)

lblThetaSAR = tk.Label(frame2, text ='θ int [deg]')
lblThetaSAR.grid(row=0,column=3)
lblThetaSARval = tk.Label(frame2, text =str(theta))
lblThetaSARval.grid(row=1,column=3)

lblNASAR = tk.Label(frame2, text ='N int')
lblNASAR.grid(row=0,column=4)
lblNASARval = tk.Label(frame2, text =str(N_A))
lblNASARval.grid(row=1,column=4)

lbldASAR = tk.Label(frame2, text ='dθ [deg]')
lbldASAR.grid(row=0,column=5)
lbldASARval = tk.Label(frame2, text =str(delta_azi))
lbldASARval.grid(row=1,column=5)

lbldrSAR = tk.Label(frame2, text ='dr [m]')
lbldrSAR.grid(row=0,column=6)
lbldrSARval = tk.Label(frame2, text =str(dr))
lbldrSARval.grid(row=1,column=6)

lbldxSAR = tk.Label(frame2, text ='dx [m]')
lbldxSAR.grid(row=0,column=7)
lbldxSARval = tk.Label(frame2, text =str(dx))
lbldxSARval.grid(row=1,column=7)

lblDSAR = tk.Label(frame2, text ='D [m]')
lblDSAR.grid(row=0,column=8)
lblDSARval = tk.Label(frame2, text =str(D))
lblDSARval.grid(row=1,column=8)


root.mainloop()