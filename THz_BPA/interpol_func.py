import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

from numba import jit, prange
@jit(nopython=True)
def interpoler(freq_set, målt, interpol_start, interpol_slut):
    
    pos=målt[3,0,:]
    målt=målt[0:3,:,:]

    
    freq_act=målt[1,:,:]
    i_ph=målt[2,:,:]
    
    #GPU_freq_set=cp.asarray(freq_set)

    #interpolation--------------------------------------------------
    i_ph_int=np.zeros((freq_set.size,i_ph[0,:].size))###f
    #GPU_i_ph_int=cp.zeros((freq_set.size,i_ph[0,:].size))
    ###f

    for io in range(0,i_ph_int[0,:].size):
        freq_til_intp=np.zeros(freq_act[:,0].size)
        freq_til_intp[0]=freq_act[0,io]
        i_ph_til_intp=np.zeros(i_ph[:,0].size)
        i_ph_til_intp[0]=i_ph[0,io]
        count=1;
        for i in range(1,freq_act[:,0].size):
            if freq_act[i,io]-freq_act[i-1,io]>0.001 and freq_act[i,io]-freq_til_intp[count-1]>0:
                freq_til_intp[count]=freq_act[i,io]
                i_ph_til_intp[count]=i_ph[i,io]        
                count=count+1
        freq_til_intp=freq_til_intp[0:count]
        i_ph_til_intp=i_ph_til_intp[0:count]


        
        #GPU_freq_til_intp=cp.asarray(freq_til_intp)
        #GPU_i_ph_til_intp=cp.asarray(i_ph_til_intp)
        #cs=CubicSpline(freq_til_intp[100:int(freq_til_intp.size-10)], i_ph_til_intp[100:int(i_ph_til_intp.size-10)])
        i_ph_int[:,io]=np.interp(freq_set, freq_til_intp, i_ph_til_intp)
        
    #i_ph_int=cp.asnumpy(GPU_i_ph_int)    
    #f12 = plt.figure()
    #ax1 = f12.add_subplot(111)
    #ax1.set_title('rå')
    #ax1.plot(freq_act[:,0],i_ph[:,0])
    #plt.xlabel("Frequency (GHz)")
    #f1 = plt.figure()
    #ax1 = f1.add_subplot(111)
    #ax1.set_title('interpoleret')
    #ax1.plot(freq_set,i_ph_int[:,0])
    return i_ph_int, pos, #f12, f1 