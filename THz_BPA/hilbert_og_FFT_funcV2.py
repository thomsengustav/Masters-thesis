import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp
import matplotlib.pyplot as plt
import time
import cupy as cp
from cupyx.scipy.signal import hilbert as gpuhilbert
def hilbert_og_fft(i_ph_int, freq_set, df, c, custom_map_linlog, phase_correct, upsampling, plot, bidder, pos_trunc_start, pos_trunc_slut):
    antal=i_ph_int[0,:].size
    i_ph_tid=np.zeros((int(i_ph_int[:,0].size*(upsampling)/2),antal), dtype=np.complex128)
    for i in range(0,bidder):
        
        et=int(i*antal/bidder)
        to=int((i+1)*antal/bidder)
       
        GPU_i_ph_int=cp.asarray(i_ph_int[:,et:to])    
        GPU_i_ph_analy=gpuhilbert(GPU_i_ph_int, axis=0)
        
        
        
        GPU_i_ph_tid=cp.fft.ifft(GPU_i_ph_analy,GPU_i_ph_analy[:,0].size*(upsampling),axis=0)
        
        #phase_corrector=np.zeros(int(i_ph_tid[:,0].size), dtype=np.complex_)
        #if phase_correct==1:
         #   for yu in range(0,int(i_ph_tid[:,0].size)):
          #      phase_corrector=np.exp(-1j*(int(i_ph_tid[:,0].size)-1)*np.pi*yu/int(i_ph_tid[:,0].size))
           # i_ph_tid=np.multiply(i_ph_tid, phase_corrector)
            
        
        #i_ph_tid=np.flip(i_ph_tid)
        

        
        N=GPU_i_ph_tid[:,0].size
        
        GPU_i_ph_tid=GPU_i_ph_tid*N/2
        
        dt=1/(N*df)
        d_dist=dt*c/2
        tid_vec=np.linspace(0, N-1,N)*dt
        distance=tid_vec*c/2
        distance=distance[0:int(np.floor(GPU_i_ph_tid[:,0].size/2))]
        
        GPU_i_ph_tid=GPU_i_ph_tid[int(cp.ceil(GPU_i_ph_tid[:,0].size/2)):GPU_i_ph_tid[:,0].size,:]
        i_ph_tid[:,et:to]=cp.asnumpy(GPU_i_ph_tid)
      
    distance=distance[int(pos_trunc_start/d_dist):int(pos_trunc_slut/d_dist)]
    i_ph_tid=i_ph_tid[int(pos_trunc_start/d_dist):int(pos_trunc_slut/d_dist)]
    
    
    
    
    


    if plot==1:
        f2 = plt.figure()
        ax2 = f2.add_subplot(111)
        ax2.set_title('enkel')
        ax2.set_xlim(3.45,3.49)
        ax2.plot(distance,np.real(i_ph_tid[:,0]))

        xv=np.linspace(0,i_ph_tid[0,:].size-1,num=i_ph_tid[0,:].size)

        fig, axa = plt.subplots()
        axa.set_title('range profile amplitude')
        axa.set_xlim(3.40,3.5)
        fp=axa.pcolormesh(distance, xv, np.transpose(np.absolute(i_ph_tid[0:i_ph_tid[:,1].size-1,0:i_ph_tid[1,:].size-1])), cmap=custom_map_linlog)
        cbar=fig.colorbar(fp,ax=axa)
        plt.xlabel("Range (m)")
        plt.ylabel("Azimuth step")
        cbar.set_label("Phase")

        fig, axa = plt.subplots()
        axa.set_title('range profile phase')
        #axa.set_xlim(3.4,3.5)
        fp2=axa.pcolormesh(distance, xv, np.transpose(np.angle(i_ph_tid[0:i_ph_tid[:,1].size-1,0:i_ph_tid[1,:].size-1]/np.absolute(i_ph_tid[0:i_ph_tid[:,1].size-1,0:i_ph_tid[1,:].size-1]))))
        fig.colorbar(fp2,ax=axa)
        plt.xlabel("Range (m)")
        plt.ylabel("Azimuth step")
        cbar.set_label("Photocurrent (nA)")
    del GPU_i_ph_int
    del GPU_i_ph_analy
    del GPU_i_ph_tid
    cp._default_memory_pool.free_all_blocks()
        
    
    return i_ph_tid, distance, dt, tid_vec, d_dist

    
    





