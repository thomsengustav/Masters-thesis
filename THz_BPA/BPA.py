# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:22:11 2025

@author: mcarl
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 21:19:34 2025

@author: mcarl
"""

from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors 
from matplotlib import colormaps
custom_map_linlog = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#000000", "#3F3F3F","#737373","#989898","#B3B3B3","#C8C8C8","#D8D8D8","#E6E6E6","#F3F3F3","#FFFFFF"])
from numba import jit
from matplotlib import cm
from celluloid import Camera
import time
from scipy.signal import find_peaks
from scipy import signal

afstandauto=0

c=3*10**8
df=0.02*10**9
plot=0


målt=np.load('20250429h12_3xBoat.npy')#np.load('20250407h15_kuglekrydsstor.npy')
#målt=np.load('20241024_kuglekryds.npy')
#målt=målt[:,0::25,:]
#målt=målt[:,:,10:200]

interpol_start, interpol_slut=430,930
BW=interpol_slut-interpol_start

freq_set=np.linspace(interpol_start, interpol_slut, int((BW*10**9/df)))

center_freq=(interpol_start)*10**9
k=2*np.pi*center_freq/c

########################## INTERPOLATIONS HJØRNET####################################
from interpol_func import interpoler
i_ph_int, pos=interpoler(freq_set, målt, interpol_start, interpol_slut)#f12, f2,
# from scipy import signal
# for i in range(0,i_ph_int[0,:].size):
#      i_ph_int[:,i]=signal.fftconvolve(i_ph_int[:,i],np.ones(3000), mode='same')
#      #i_ph_int[:,i]=signal.fftconvolve(i_ph_int[:,i],np.ones(100), mode='same')

f12 = plt.figure()
ax1 = f12.add_subplot(111)
ax1.set_title('rå')
ax1.plot(målt[1,:,0],målt[2,:,0])
plt.xlabel("Frequency (GHz)")
f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.set_title('interpoleret')
ax1.plot(freq_set,i_ph_int[:,0])
print('Den er interpoleret')
#####################################################################################
#####################HILBERT###########
phase_correct=0
upsampling=16
bidder=10

pos_trunc_start=3.35
pos_trunc_slut=3.6

tidtager=time.time()
from hilbert_og_FFT_funcV2 import hilbert_og_fft
i_ph_tid, distance, dt, tid_vec, d_dist=hilbert_og_fft(i_ph_int, freq_set, df, c, custom_map_linlog, phase_correct, upsampling, plot, bidder, pos_trunc_start, pos_trunc_slut)

print('Så er der taget Hilbert og IFFT')  
print(time.time()-tidtager)
#####################################################################
tester=i_ph_tid[:,30]
f224 = plt.figure()
ax224 = f224.add_subplot(111)
ax224.set_title('uden')
ax224.plot(distance,np.abs(tester))
ax224.set_xlim(3.46,3.58)
ax224.set_ylim([0,150])


testerfft=np.fft.fft(tester)

freq2=np.linspace(interpol_start, interpol_slut,num=int(testerfft.size/upsampling))
freq3=np.zeros(testerfft.size)
freq3[0:int(testerfft.size/upsampling)]=freq2*10**9


freq2=np.linspace(interpol_start, interpol_slut,num=int(testerfft.size/upsampling))
freq3=np.zeros(testerfft.size)
freq3[0:int(testerfft.size/upsampling)]=freq2*10**9

f224 = plt.figure()
ax224 = f224.add_subplot(111)
ax224.set_title('enkel med filter')
ax224.plot(freq3,np.abs(testerfft))


dt2=0.00245#135*dt/2*c
vægt=0.5*0
vægt2=0.15#-0.5#-0.2

dt22=0.0355#0.00245#135*dt/2*c
vægt1=-0.03#0.15
vægt12=0.15*0#-0.5#-0.2
testerfftcoor=testerfft/(1+vægt*np.exp(-1j*np.pi*dt2/c*4*freq3)+vægt2*np.exp(1j*np.pi*dt2/c*4*freq3)+vægt1*np.exp(-1j*np.pi*dt22/c*4*freq3)+vægt12*np.exp(1j*np.pi*dt22/c*4*freq3))
testercoor=np.fft.ifft(testerfftcoor)


f224 = plt.figure()
ax224 = f224.add_subplot(111)
ax224.set_title('med')
ax224.plot(distance,np.abs(testercoor))
ax224.set_xlim(3.46,3.6)
ax224.set_ylim([0,60])
####################################################################
pulsfixer=0
if pulsfixer==1:
    for i in range(0,i_ph_tid[0,:].size):
        testerfft=np.fft.fft(i_ph_tid[:,i])
        testerfftcoor=testerfft/(1+vægt*np.exp(-1j*np.pi*dt2/c*4*freq3)+vægt2*np.exp(1j*np.pi*dt2/c*4*freq3)+vægt1*np.exp(-1j*np.pi*dt22/c*4*freq3)+vægt12*np.exp(1j*np.pi*dt22/c*4*freq3))
        testercoor=np.fft.ifft(testerfftcoor)
           
        i_ph_tid[:,i]=testercoor
#####################################################################

#ref_puls=np.load('ref_puls8samp750um500ghz.npy')

from rangecomp_func import range_compression
range_compressed=i_ph_tid#range_compression(i_ph_tid, c, dt, ref_puls, plot)
#range_compressed=range_compression(i_ph_tid, c, dt, ref_puls, plot)
legmedfreq=1
filter1=1
filter2=0
filter3=0
if legmedfreq==1:
    GPU_range_compressed=cp.asarray(range_compressed)
    GPU_freqdomain=cp.fft.fft(GPU_range_compressed,axis=0)
    freqdomain=cp.asnumpy(GPU_freqdomain)
    del GPU_range_compressed

    cutoff=int(np.ceil((pos_trunc_slut-pos_trunc_start)/3.75/2*i_ph_int[:,0].size))
    
    a0=0.5
    window=np.zeros(cutoff)
    for i in range(0,cutoff):
        window[i]=a0*(1-np.cos(2*np.pi*i/(cutoff)))
    window2=np.zeros(cutoff)    
    
    freqdomain_abs_sum=np.absolute(freqdomain[0:cutoff])
    freqdomain_abs_sum=np.sum(freqdomain_abs_sum,axis=1)
    freqdomain_abs_sum=np.convolve(freqdomain_abs_sum,np.ones(20),mode='same')
    freqdomain_abs_sum=freqdomain_abs_sum/np.max(freqdomain_abs_sum)
    
    freq_corrector=np.zeros(range_compressed[:,0].size)
    freq_corrector[0:cutoff]=1/(freqdomain_abs_sum)
    if filter2==1:
        freq_corrector=np.ones(range_compressed[:,0].size)
        
    if filter3==1:
        window = signal.windows.taylor(cutoff, nbar=40, sll=65, norm=False)
        freq_corrector[0:cutoff]=freq_corrector[0:cutoff]*window
        
    if filter1==1:
        freq_corrector[0:cutoff]=freq_corrector[0:cutoff]*window
    
    GPU_freq_corrector=cp.asarray(freq_corrector)
    GPU_freq_corrector=cp.tile(GPU_freq_corrector,i_ph_tid[0,:].size)
    GPU_freq_corrector=cp.transpose(cp.reshape(GPU_freq_corrector, (range_compressed[0,:].size,range_compressed[:,0].size)))
    GPU_range_compressed=cp.fft.ifft(GPU_freqdomain*GPU_freq_corrector,axis=0)
    range_compressed=cp.asnumpy(GPU_range_compressed)
    del GPU_range_compressed, GPU_freq_corrector, GPU_freqdomain


    
      

    aa,bb=3.35,3.6
    f24 = plt.figure()
    ax24 = f24.add_subplot(111)
    ax24.set_title('enkel')
    ax24.plot(distance,20*np.log10(np.abs(np.fft.ifft(freqdomain[:,30])/np.max(np.abs(np.fft.ifft(freqdomain[:,30]))))))
    ax24.plot(distance,20*np.log10(np.abs(range_compressed[:,30])/np.max(np.abs(range_compressed[:,30]))))
    ax24.set_xlim(aa,bb)
    ax24.set_ylim([-40,0])
    f224 = plt.figure()
    ax224 = f224.add_subplot(111)
    ax224.set_title('enkel med filter')
    ax224.plot(distance,20*np.log10(np.abs(range_compressed[:,30])/np.max(np.abs(range_compressed[:,30]))))
    ax224.set_xlim(aa,bb)
    ax224.set_ylim([-40,0])

xv=np.linspace(0,i_ph_tid[0,:].size-1,num=i_ph_tid[0,:].size)

fig, axa1 = plt.subplots()
axa1.set_title('range profile amplitude')
axa1.set_xlim(3.45,3.58)
fp1=axa1.pcolormesh(distance, xv, np.transpose(np.absolute(range_compressed[0:range_compressed[:,1].size-1,0:range_compressed[1,:].size-1])), cmap=custom_map_linlog)#, vmax=700)
cbar=fig.colorbar(fp1,ax=axa1)
plt.xlabel("Range (m)")
plt.ylabel("Azimuth step")
cbar.set_label("Phase")

fig, axa1 = plt.subplots()
axa1.set_title('range profile amplitude')
axa1.set_xlim(3.45,3.58)
fp1=axa1.pcolormesh(distance, xv, np.transpose(np.angle(range_compressed[0:range_compressed[:,1].size-1,0:range_compressed[1,:].size-1])))#, cmap=custom_map_linlog)#, vmax=700)
cbar=fig.colorbar(fp1,ax=axa1)
plt.xlabel("Range (m)")
plt.ylabel("Azimuth step")
cbar.set_label("Phase")


########################POSITION#####################################
theta=(10.2)*np.pi/180 #(10.2)*np.pi/180 #12.65#11.69 17.95 bil skrå +20.7, godfin 10.2, 
d_em=0.85017#0.804487##afstand fra emitter til center of rotation
d_dec=0.821671#0.800032#0.800558##afstand fra detector til center of rotation
d_bistatic_angle=2*np.arctan(4.5/80)#bistatic vinkel 6.375
phi_v=-pos*np.pi/180+pos[-1]/2*np.pi/180#rotation af sample
#phi_v=phi_v*0.8


##########################################################################

x_vec=np.linspace(-0.02, 0.02, num=500, dtype=np.float64)
y_vec=np.linspace(-0.02, 0.02, num=500, dtype=np.float64)


from Billede_danner_GPU_bistatic import Billede_danner_GPU_bistatic
from position import  Positionsregner
start_pos=2.64944#5395#2.66396#2.66396



    


    

x_opdel, y_opdel=5,5
xSize,ySize=x_vec.size, y_vec.size
I=np.zeros((xSize,ySize), dtype=np.complex128)
tidtager=time.time()


if afstandauto==1:
    def regnE(x_opdel, y_opdel, x_vec, y_vec, xSize,ySize,I, nul1, d_dist, k, range_compressed,d_em, d_dec):
        
        pos_emitter, pos_detector=Positionsregner(d_em, d_dec, phi_v, d_bistatic_angle, theta)
        for x in range(0,x_opdel):
             xV=x_vec[int(xSize/x_opdel*x):int(xSize/x_opdel*(x+1))]
             for y in range(0,y_opdel):
                 yV=-1*y_vec[int(ySize/y_opdel*y):int(ySize/y_opdel*(y+1))]
                 
                 I2=Billede_danner_GPU_bistatic(yV,xV, pos_emitter, pos_detector, nul1, d_dist, k, range_compressed)
                 I[int(xSize/x_opdel*x):int(xSize/x_opdel*(x+1)),int(ySize/y_opdel*y):int(ySize/y_opdel*(y+1))]=I2
                    
        SUM_I=np.sum(np.absolute(I)**2)
        I_norm=np.absolute(I)**2/SUM_I
        E=-np.sum(np.multiply(I_norm,np.log(I_norm)))
            #print(E)
        return E
    corrector=np.asarray([0.80,0.80,start_pos])
    from scipy.optimize import minimize
    def minimizer(corrector):
        nul1=+int(corrector[2]/d_dist)-int(pos_trunc_start/d_dist)
        d_em, d_dec=corrector[0], corrector[1]
        E=regnE(x_opdel, y_opdel, x_vec, -1*y_vec, xSize,ySize,I, nul1, d_dist, k, range_compressed,d_em, d_dec)
        print(E)
        return E

    bound_corrector=np.zeros([corrector.size,2])
    bound_corrector[0:2,0]=0.82#0.797#0.79
    bound_corrector[0:2,1]=0.86#0.802#0.81
    bound_corrector[2,0]=2.62#2.66
    bound_corrector[2,1]=2.68#2.67


    res = minimize(minimizer, corrector, method='Powell', bounds=bound_corrector, options={'maxiter':2})#Powell
    corrector2=res.x
    
else:
    corrector2=np.array([d_em,d_dec,start_pos])#2.66398



from Billede_danner_GPU_bistatic_V2 import Billede_danner_GPU_bistatic_V2       

print(time.time()-tidtager)
I_puls_tot=np.zeros((xSize*ySize,range_compressed[0,:].size),dtype=np.complex128)
azi_v_tot=np.zeros((xSize*ySize,range_compressed[0,:].size),dtype=np.complex128)
def regnI(x_opdel, y_opdel, x_vec, y_vec, xSize,ySize,I, nul1, d_dist, k, range_compressed,d_em, d_dec):
    count=0;
    pos_emitter, pos_detector=Positionsregner(d_em, d_dec, phi_v, d_bistatic_angle, theta)
    for x in range(0,x_opdel):
         xV=x_vec[int(xSize/x_opdel*x):int(xSize/x_opdel*(x+1))]
         for y in range(0,y_opdel):
             yV=-1*y_vec[int(ySize/y_opdel*y):int(ySize/y_opdel*(y+1))]
             
             I2,I_puls,azi_v=Billede_danner_GPU_bistatic_V2(yV,xV, pos_emitter, pos_detector, nul1, d_dist, k, range_compressed)
             I[int(xSize/x_opdel*x):int(xSize/x_opdel*(x+1)),int(ySize/y_opdel*y):int(ySize/y_opdel*(y+1))]=I2
             I_puls_tot[int(ySize/y_opdel*xSize/x_opdel*count):int(ySize/y_opdel*xSize/x_opdel*(count+1)),:]=I_puls
             azi_v_tot[int(ySize/y_opdel*xSize/x_opdel*count):int(ySize/y_opdel*xSize/x_opdel*(count+1)),:]=azi_v
             count+=1
    SUM_I=np.sum(np.absolute(I)**2)
    I_norm=np.absolute(I)**2/SUM_I
    E=-np.sum(np.multiply(I_norm,np.log(I_norm)))
    print('Entro')
    print(E)
    return I,I_puls_tot,azi_v

nul1=int(corrector2[2]/d_dist)-int(pos_trunc_start/d_dist)
d_em, d_dec=corrector2[0], corrector2[1]
I=np.zeros((xSize,ySize), dtype=np.complex128)
I,I_puls_tot, azi_v=regnI(x_opdel, y_opdel, x_vec, y_vec, xSize,ySize,I, nul1, d_dist, k, range_compressed,d_em, d_dec)


I=np.transpose(I)
#I=np.flip(I, axis=0)
#I=I/np.max(np.abs(I))
fig, axa = plt.subplots()
axa.set_title('',fontsize=16)
fpf=axa.pcolormesh(x_vec*100, y_vec*100, np.abs(I[0:I[:,1].size-1,0:I[1,:].size-1]), cmap=custom_map_linlog)#,vmax=500,vmin=80)#, vmax=10000)#, vmax=18000)#, vmax=60000)#, vmax=60000)#,vmin=0, vmax=500)
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [cm]",fontsize=16)
plt.ylabel("Azimuth [cm]",fontsize=16)
cbar.set_label("Intensity",fontsize=14)
cbar.set_ticks([])
axa.set_aspect('equal')
#plt.savefig('yacht_full_linlog.png', dpi=250, bbox_inches='tight')


I_DB=20*np.log10(np.abs(I)/np.max(np.abs(I)))

import matplotlib.ticker as ticker
fig, axa = plt.subplots()
axa.set_title('200 $\mu$m width',fontsize=16)
fpf=axa.pcolormesh(x_vec*1000, y_vec*1000, (I_DB[0:I_DB[:,1].size-1,0:I_DB[1,:].size-1]),cmap='gray',vmin=-20)#, vmax=60000)#,vmin=0, vmax=500)
cbar=fig.colorbar(fpf,ax=axa)
#axa.set_xlim([5.5,7])
#axa.set_ylim([-3,-1.5])
plt.xlabel("Range [mm]",fontsize=16)
plt.ylabel("Azimuth [mm]",fontsize=16)
cbar.set_label("Normalized Intensity [dB]",fontsize=14)
axa.set_aspect('equal')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
axa.xaxis.set_major_locator(ticker.LinearLocator(3))
axa.yaxis.set_major_locator(ticker.LinearLocator(5))
#plt.savefig('yacht_full.png', dpi=250, bbox_inches='tight')

