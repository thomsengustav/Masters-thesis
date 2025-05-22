from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt
import numpy as np

from scipy import io, integrate, linalg, signal
from scipy.signal import hilbert, chirp
from scipy.sparse.linalg import cg, eigs
from scipy.special import jv, yv
from matplotlib.colors import LogNorm
import pandas as pd
#import-----------------------------
df=0.02
input  = pd.read_table("C:\\Users\\detri\\Desktop\\LMG THz Microscopy\\data\\hej", sep='\s+')
input_fin=np.array(input)
freq_set=input_fin[:,0]
i_ph=input_fin[:,1]
freq_act=input_fin[:,2]
del input, input_fin

f1 = plt.figure()
ax1 = f1.add_subplot(111)
ax1.set_title('rå')
ax1.plot(freq_set,i_ph)
#interpolation--------------------------------------------------
freq_til_intp=np.zeros(freq_act.size)
freq_til_intp[0]=freq_act[0]
i_ph_til_intp=np.zeros(i_ph.size)
i_ph_til_intp[0]=i_ph[0]
count=1;
for i in range(1,freq_act.size):
    if freq_act[i]-freq_act[i-1]>0:
        freq_til_intp[count]=freq_act[i]
        i_ph_til_intp[count]=i_ph[i]        
        count=count+1
freq_til_intp=freq_til_intp[0:count]
i_ph_til_intp=i_ph_til_intp[0:count]

freq_set=np.linspace(500, 1500, int(freq_set.size))

i_ph_int=np.interp(freq_set, freq_til_intp, i_ph_til_intp, left=None, right=None)
del freq_til_intp, i_ph_til_intp, count
f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.set_title('interpoleret')
ax2.plot(freq_set,i_ph_int)
#hilbert--------------------------------------------------------------
i_ph_analy=np.array(hilbert(i_ph_int))
f2 = plt.figure()
ax2 = f2.add_subplot(111)
ax2.set_title('hilbert envelope')
ax2.plot(freq_set,np.absolute(i_ph_analy))


    
#filterr=np.flip(filterr)



#Fourier fra freq til tids domæne
i_ph_tid=np.fft.ifft(i_ph_analy)
#i_ph_tid=np.flip(i_ph_tid)

N=i_ph_tid.size
dt=1/(N*df)
tid_vec=np.linspace(0, N-1,N)*dt
distance=tid_vec*10**-9*3*10**8/2

f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.set_title('i_ph_tid')
ax3.set_xlim(1.5,2)
ax3.plot(distance,np.real(i_ph_tid))

f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.set_title('i_ph_tid')
ax3.set_xlim(7,7.5)
#ax3.set_ylim(-0.005,0.005)
ax3.plot(distance,np.real(i_ph_tid))


#laver range profile
azi_step=100
azi_v=np.zeros(azi_step+1)
angle_profile=np.zeros((azi_step,i_ph_tid.size))
range_profile=np.zeros((azi_step,i_ph_tid.size))
for i in range(0,azi_step):
    range_profile[i,:]=np.absolute(i_ph_tid)
    angle_profile[i,:]=np.angle(i_ph_tid)
    azi_v[i]=i

distance2=distance[0:5001]
range_profile2=range_profile[:,0:5000]
fig, ax = plt.subplots()
ax.set_title('range profile')
ax.pcolormesh(distance2 ,azi_v, range_profile2)




angle_profile2=angle_profile[:,0:2000]

fig, axa = plt.subplots()
axa.set_title('angle profile')
axa.pcolormesh(distance2 ,azi_v, angle_profile2)

#tester
test1=i_ph_tid[750:1300]
f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.set_title('snit')
ax3.plot(test1)
