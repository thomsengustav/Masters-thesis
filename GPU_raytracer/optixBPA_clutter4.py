from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from scipy import io, integrate, linalg, signal
import matplotlib as mpl
import matplotlib.colors 
import time
custom_map_linlog = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#000000", "#3F3F3F","#737373","#989898","#B3B3B3","#C8C8C8","#D8D8D8","#E6E6E6","#F3F3F3","#FFFFFF"])

plot=0
#rangeV=np.loadtxt("C:\\Users\\mcarl\\OneDrive\\Dokumenter\\KandidatSpeciale\\optixBPA\\Mstar_res\\test4ny\\C:/ProgramData/NVIDIA Corporation/OptiX SDK 8.0.0/SDK/optixTriangle/datamappen/-92.000000MSTAR_radius_tjek_80r_spherical.txt.txt")
rangeV=np.loadtxt("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixTriangle\\datamappen\\-92.000000MSTAR_radius_tjek_80r_spherical.txt")#+np.loadtxt("C:\\ProgramData\\NVIDIA Corporation\\OptiX SDK 8.0.0\\SDK\\optixTriangle\\datamappen\\yacht2.txt")#10:tankkugleplan
rangeV=rangeV.astype(np.complex128)
#rangeV=rangeV[0:int(174*80000)]
binsize=0.016/4
antalmål=153
indgange=50000

rangeV=rangeV[0:antalmål*indgange]
nr=np.linspace(0,antalmål-1,antalmål)


radius=40*10#20*1.4
nul1=-900-9000-radius
slant=15*np.pi/180
radar_range=250*2+4500
vinkelint=2.94

range_vec=np.linspace(0, indgange*binsize, indgange)
#etter=np.ones(5)
#rangeV=signal.fftconvolve(rangeV, etter, mode='same')


f52 = plt.figure()
ax52 = f52.add_subplot(111)
ax52.set_title('rangeV')
ax52.plot(rangeV[0:indgange])
#ax52.set_xlim([10000+0,10000+120])
rangeV2=np.zeros(rangeV.size, dtype=np.complex128)

rangeV2=np.abs(rangeV)**2*np.sign(rangeV)#**0.5#+0.5*rangeV[:,2]*1j

ff=np.isnan(rangeV2)
fff=np.asarray(np.where(ff==1))
ffi=np.isinf(rangeV2)
fffi=np.asarray(np.where(ffi==1))
print(fffi.sum())
print(ff.sum())

# a=fff[0,0]

# print(rangeV2[fff[0,0]])
# print(rangeV[1,0])
# print(rangeV[a-1,1])
# print(rangeV[a-1,2])

print(rangeV[fff[0]])
rangeV2[fff[0]]=0;
rangeV2[fffi[0]]=0;

f52 = plt.figure()
ax52 = f52.add_subplot(111)
ax52.set_title('rangeV2')
ax52.plot(np.abs(rangeV2[0:indgange]))

rangeM=rangeV2.reshape((antalmål,indgange))
#rangeM[np.where(rangeM>40)]=40

del rangeV
del rangeV2

f5 = plt.figure()
ax5 = f5.add_subplot(111)
ax5.set_title('rangeM')
ax5.plot(range_vec,rangeM[0,:])
ax5.set_xlim([1090,1170])


#########################Noise rangeM################################
# xGauss=np.linspace(-0.7,0.7,)
# Gauss_sig=0.5
# Gauss=1/(Gauss_sig*np.sqrt(2*np.pi))*np.exp(-xGauss**2/(2*Gauss_sig**2))
# f5 = plt.figure()
# ax5 = f5.add_subplot(111)
# ax5.plot(Gauss)
# X_multi=np.zeros((rangeM[:,0].size,rangeM[0,:].size), dtype=np.complex128)
# for i in range(44,rangeM[0,:].size-44):
#     for j in range(0,rangeM[:,0].size):
#         X_multi[j,i-44:i+44]+=rangeM[j,i-44:i+44]+rangeM[j,i]*np.random.rand(88)*2

# f5 = plt.figure()
# ax5 = f5.add_subplot(111)
# ax5.set_title('Multi')
# ax5.plot(range_vec,X_multi[0,:])
# ax5.set_xlim([1100,1125])
# rangeM=X_multi
#####################################################################


rangeM2=rangeM[:,64700:65500]
if plot==1:
    fig, axa = plt.subplots()
    axa.set_title('rangeM')
    axa.pcolormesh(np.abs(rangeM2))
    #axa.set_xlim([180,320])
    fig, axa = plt.subplots()
    axa.set_title('rangeM')
    axa.plot(np.abs(rangeM2[0,:]))
    axa.set_xlim([3000,4000])


antal_hits=np.sum(rangeM)/antalmål

########################ovenfor er import af optix###############################
#chirp signal
T=5*10**-8
f0=9.6*10**9
B_c=0.591*10**9
K=B_c/T

c=3*10**8
k=2*np.pi*f0/c
dt=binsize/c
tn=0
tid_enkel=np.linspace(-0.6*T, 0.6*T, num=int(1.2*T/dt))
enkel_puls=np.zeros(tid_enkel.size, dtype=np.complex_)
for i in range(0,tid_enkel.size):
    t=tid_enkel[i]
    x1=(T)/2-((t-tn)**2)**0.5
    enkel_puls[i]=np.heaviside(x1, 1)*np.exp(1j*(2*np.pi*f0*(t-tn)+np.pi*K*(t-tn)**2))
fig, axa2 = plt.subplots()
axa2.set_title('enkel ikke moduleret')    
axa2.plot(tid_enkel,np.real(enkel_puls))
#axa.set_xlim([-5*10**-8,-4.9*10**-8])

t1_vec=np.linspace(0, dt*(indgange-1), num=indgange)
modulator=np.zeros(t1_vec.size, dtype=np.complex_)
for i in range(0,t1_vec.size):
    modulator[i]=np.exp(-1j*2*np.pi*f0*t1_vec[i])


#############################NOISE####################################
#Xamp=np.zeros(10000)
#Xphase=np.zeros(10000)
average_Noise=0
varians=-10

Xamp=cp.random.normal(loc=average_Noise, scale=10**(varians/20), size=antalmål*indgange).reshape(antalmål,indgange)
Xphase=cp.random.rand(antalmål,indgange)*2*cp.pi

Y_additive_noise=cp.asnumpy(Xamp)#*np.exp(1j*Xphase))
del Xamp, Xphase
  
plt.hist(np.abs(Y_additive_noise[0,:]), bins='auto')
plt.show()


######################################################################
rangeM_signal=np.zeros((antalmål, indgange), dtype=np.complex128)
for i in range(0,antalmål):
   
    rangeM_signal[i,:]=(signal.fftconvolve(rangeM[i,:], enkel_puls,'same'))*modulator
fig, axa3 = plt.subplots()
axa3.set_title('conv')    
axa3.plot(range_vec,np.real(rangeM_signal[0,:]))
#rangeM_signal=np.abs(rangeM_signal)**0.5*np.exp(1j*np.angle(rangeM_signal))
#axa3.set_xlim([1000,1200])  

# fdf=rangeM_signal[1,:]
# ttt=rangeV[:,2]
# dd=np.isnan(fdf)
# print(dd.sum())

# dd=np.isnan(ttt)
# print(dd.sum())

# dd=np.isnan(rangeM_signal)
# print(dd.sum())

if plot==1:
    fig, axa4 = plt.subplots()
    axa4.set_title('rangeM signal')
    axa4.pcolormesh(range_vec,nr,np.abs(rangeM_signal))
    #axa4.set_xlim([60,320]) 
 
#########################nedenfor ligesom RT BPA#############################

range_prof=rangeM_signal
azi_v=np.linspace(0,vinkelint*(antalmål-1)/antalmål,antalmål)*np.pi/180-vinkelint/2*np.pi/180+np.pi/2 +180*np.pi/180

#azi_v=np.linspace(vinkelint*(antalmål-1)/antalmål,0,antalmål)*np.pi/180-vinkelint/2*np.pi/180+np.pi/2 +180*np.pi/180
#azi_v=azi_v*0.75


R_Mat=np.zeros((azi_v.size,3))



R_Mat[:,2]=np.sin(slant)*radar_range
R_Mat[:,0]=np.cos(azi_v)*np.cos(slant)*radar_range
R_Mat[:,1]=np.sin(azi_v)*np.cos(slant)*radar_range

sidste_t_vec=range_vec/c
###########################################

#laver enkel chirp puls som skal bruges i range compression 
tid_enkel=np.linspace(-0.6*T, 0.6*T, num=int(1.2*T/dt))
range_ref=np.zeros(tid_enkel.size, dtype=np.complex_)
for i in range(0,tid_enkel.size):
    t=tid_enkel[i]
    x1=(T)/2-((t-0)**2)**0.5
    range_ref[i]=np.heaviside(x1, 1)*np.exp(1j*(np.pi*K*(t-0)**2))
f3= plt.figure()
ax3 = f3.add_subplot(111)
ax3.set_title('range_ref')
ax3.plot(tid_enkel,range_ref)



#Range compression------------------------------
t2_vec=np.linspace(tid_enkel[0]+sidste_t_vec[0], tid_enkel[-1]+sidste_t_vec[-1], num=tid_enkel.size+sidste_t_vec.size-1)
#range_compressed=np.zeros((R_Mat[:,0].size, t2_vec.size), dtype=np.complex_)
range_compressed=np.zeros((R_Mat[:,0].size, sidste_t_vec.size), dtype=np.complex_)
for i in range(0,range_compressed[:,0].size):
    range_compressed[i,:]=signal.fftconvolve(range_prof[i,:], np.conjugate(range_ref), mode='same')*dt

distance3=sidste_t_vec*c/2
azi=R_Mat[:,0]
#range_compressed2=np.transpose(range_compressed)

range_compressed2=range_compressed[0:azi.size-1,0:sidste_t_vec.size-1]
range_compressed2=np.flip(range_compressed2,axis=0)

if plot==1:
    fig, axa = plt.subplots()
    axa.set_title('range compressed amplitude')
    axa.pcolormesh(distance3, azi, np.absolute(range_compressed2))
    #axa.set_xlim([30,70])
    
    #axa.set_xlim([1000,1200])

#Til billede--------------------------------------------------------------------------
x_vec=np.linspace(-13.8, 13.8, num=138*1) #Scene center er sat i (x=0,y=0)
y_vec=np.linspace(-13.8, 13.8, num=138*1)
#x_vec=np.linspace(-5, 5, num=138) #Scene center er sat i (x=0,y=0)
#y_vec=np.linspace(-5, 5, num=138)  #x og y interval i meter, num=antal pixels i x og y dim
I_puls=np.zeros([azi_v.size], dtype=np.complex_)
I=np.zeros([x_vec.size,y_vec.size], dtype=np.complex_)#matrice med pixel værdier

#position af emitter og detector, her er det sat til monostatic
pos_emitter=np.transpose(R_Mat)
pos_detector=np.zeros((3,antalmål))
for i in range(0,antalmål):
    pos_detector[:,i]=pos_emitter[:,i]+pos_emitter[:,i]/np.linalg.norm(pos_emitter[:,i])*1.1*radius
    


range_compressed=np.transpose(range_compressed)
range_compressed=np.flip(range_compressed, axis=1)

filterr=1

fig, axa = plt.subplots()
axa.set_title('range compressed amplitude')
axa.plot(distance3,np.absolute(range_compressed[:,0]))


if filterr==1:
    df=1/(indgange*dt)
    bw_ingange=int(B_c/df/2)
    window = signal.windows.taylor(bw_ingange*2, nbar=20, sll=35, norm=False)
    
    fftrange=np.fft.fft(range_compressed,axis=0)
    fig, axa = plt.subplots()
    axa.set_title('range compressed freq')
    axa.plot(np.abs(fftrange[:,0]))
    fftrange=np.fft.fftshift(fftrange,axes=0)
    fig, axa = plt.subplots()
    axa.set_title('range compressed freq')
    axa.plot(np.abs(fftrange[:,0]))
    fftrange[int(indgange/2-bw_ingange):int(indgange/2+bw_ingange)]=fftrange[int(indgange/2-bw_ingange):int(indgange/2+bw_ingange)]*window[:,None]
    fig, axa = plt.subplots()
    axa.set_title('range compressed freq')
    axa.plot(np.abs(fftrange[:,0]))
    fftrange=np.fft.fftshift(fftrange,axes=0)
    range_compressed=np.fft.ifft(fftrange,axis=0)
    



    
d_dist=dt*c/2
#nul1=-int(t2_vec[0]*c/2/d_dist)# bruges til vælge rigtige indgange i range_compressed matricen i billede danner 


#Laver et billede, cupy skal være hentet, ellers skal Billede_danner lige rettes
opdel_pulser=30

tidtager=time.time()
from Billede_danner_GPU_bistatic import Billede_danner_GPU_bistatic


for i in range(0,opdel_pulser):
    index1=int(i*antalmål/opdel_pulser)
    index2=int((i+1)*antalmål/opdel_pulser)
    I+=Billede_danner_GPU_bistatic(y_vec,x_vec, pos_emitter[:,index1:index2], pos_detector[:,index1:index2], nul1, d_dist, k, range_compressed[:,index1:index2])
    print(i)

print(time.time()-tidtager)



I=np.rot90(I, k=3, axes=(0, 1))
#I=signal.convolve2d(I, np.ones((1,2)), mode='same')
import matplotlib.ticker as ticker
fig, axa = plt.subplots()
axa.set_title('Box on clutter',fontsize=16)
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I[0:I[:,1].size-1,0:I[1,:].size-1]), cmap='gray',vmax=0.00001)#cmap=custom_map_linlog)#, vmax=0.0015)
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]",fontsize=16)
plt.ylabel("Azimuth [m]",fontsize=16)
cbar.set_label("Intensity",fontsize=14)
cbar.set_ticks([])
axa.set_aspect('equal')
#axa.xaxis.set_major_locator(ticker.LinearLocator(5))
#axa.yaxis.set_major_locator(ticker.LinearLocator(5))



I_DB=20*np.log10(np.abs(I)/np.max(np.abs(I)))

#x_vec=x_vec-2
#y_vec=y_vec+2.5
fig, axa = plt.subplots()
axa.set_title(str(radius)+"m"+" radius")
fpf=axa.pcolormesh(y_vec, x_vec, (I_DB[0:I_DB[:,1].size-1,0:I_DB[1,:].size-1]),cmap='gray',vmax=0,vmin=-30)#, vmax=60000)#, vmax=60000)#,vmin=0, vmax=500)
cbar=fig.colorbar(fpf,ax=axa)
#axa.set_xlim([-20,-15])
#axa.set_ylim([15,20])
plt.xlabel("Range [m]")
plt.ylabel("Azimuth [m]")
cbar.set_label("Normalized Intensity [dB]")
axa.set_aspect('equal')
#plt.savefig('ford_bagside_flat.png', dpi=250, bbox_inches='tight')
azfilter=1
I_az=np.zeros((x_vec.size,y_vec.size), dtype=np.complex128)
if azfilter==1:
    window_az = signal.windows.taylor(y_vec.size, nbar=20, sll=35, norm=False)
    for d in range(0,x_vec.size):
        f=I[d,:]       
        f_fft=np.fft.fftshift(np.fft.fft(f))*window_az
        I_az[d,:]=np.fft.ifft(np.fft.fftshift(f_fft))
        

Vmax=np.sum(np.abs(I_az))/(138**2)*4#*#1.5
fig, axa = plt.subplots()
axa.set_title('Pin hole',fontsize=16)
fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I_az[0:I[:,1].size-1,0:I[1,:].size-1]), cmap='gray',vmax=Vmax)#cmap=custom_map_linlog)#, vmax=0.0015)
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range [m]",fontsize=16)
plt.ylabel("Azimuth [m]",fontsize=16)
cbar.set_label("Intensity",fontsize=14)
cbar.set_ticks([])
axa.set_aspect('equal')
    




IDB4=np.transpose(I_DB)
import matplotlib.ticker as ticker
fig, axa = plt.subplots()
axa.set_title('Simulation',fontsize=16)
fpf=axa.pcolormesh(y_vec/2, x_vec/2, (IDB4[0:I_DB[:,1].size-1,0:I_DB[1,:].size-1]),cmap='gray',vmin=-30)#, vmax=60000)#,vmin=0, vmax=500)
cbar=fig.colorbar(fpf,fraction=0.046, pad=0.04)
#axa.set_xlim([5.5,7])
#axa.set_ylim([-3,-1.5])
plt.xlabel("Azimuth [mm]",fontsize=16)
plt.ylabel("Range [mm]",fontsize=16)
cbar.set_label("Normalized Intensity [dB]",fontsize=14)
axa.set_aspect('equal')
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
axa.xaxis.set_major_locator(ticker.LinearLocator(5))
axa.yaxis.set_major_locator(ticker.LinearLocator(3))
axa.set_ylim([-10,10])






forvent=np.zeros(R_Mat[:,0].size)
for i in range(0,R_Mat[:,0].size):
    forvent[i]=(np.linalg.norm([6,4,0]-R_Mat[i,:])+np.linalg.norm([6,4,0]-pos_detector[:,i]))/2-1
    






x_vec2=np.linspace(-0.2*138/2, 0.2*138/2, num=300) #Scene center er sat i (x=0,y=0)
y_vec2=np.linspace(-0.2*138/2,0.2*138/2, num=300)


I2r=I[:,0::4]+I[:,1::4]+I[:,2::4]+I[:,0::4]
I2=I2r[0::4,:]+I2r[1::4,:]+I2r[2::4,:]+I2r[3::4,:]

fig, axa = plt.subplots()
axa.set_title('image')
fpf=axa.pcolormesh(y_vec2, x_vec2, np.absolute(I2[0:I2[:,1].size-1,0:I2[1,:].size-1]), cmap=custom_map_linlog)#, vmax=0.3)
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range(x) (m)")
plt.ylabel("Azimuth(y) (m)")
cbar.set_label("Intensity (a.u.)")
axa.set_aspect('equal')


fig, axa = plt.subplots()
axa.set_title('image')
fpf=axa.pcolormesh(y_vec, x_vec, np.angle(I[0:I[:,1].size-1,0:I[1,:].size-1]), cmap='twilight_shifted')#, vmax=0.3)
cbar=fig.colorbar(fpf,ax=axa)
plt.xlabel("Range(x) (m)")
plt.ylabel("Azimuth(y) (m)")
cbar.set_label("Intensity (a.u.)")
axa.set_aspect('equal')



fig, axa = plt.subplots()
axa.set_title('range compressed amplitude')
axa.plot(distance3, 20*np.log10(np.absolute(range_compressed[:,0])))
axa.set_xlim([730,750])
axa.set_ylim([-150,-60])



ayayaya=np.fft.fftshift(np.fft.fft(range_compressed[:,0]))
fig, axa = plt.subplots()
axa.set_title('range compressed freq')
axa.plot(np.abs(ayayaya))
axa.set_xlim([640000/2-bw_ingange,640000/2+bw_ingange])

window = signal.windows.taylor(bw_ingange*2, nbar=20, sll=70, norm=False)
plt.plot(window)
plt.title("Taylor window (35 dB)")
plt.ylabel("Amplitude")
plt.xlabel("Sample")

ayayaya[int(640000/2-bw_ingange):int(640000/2+bw_ingange)]=ayayaya[int(640000/2-bw_ingange):int(640000/2+bw_ingange)]*window
fig, axa = plt.subplots()
axa.set_title('range compressed freq')
axa.plot(np.abs(ayayaya))
axa.set_xlim([640000/2-bw_ingange,640000/2+bw_ingange])

fig, axa = plt.subplots()
axa.set_title('range compressed freq')
axa.plot(distance3,20*np.log10(np.abs(np.fft.ifft(np.fft.fftshift((ayayaya))))))
axa.plot(distance3, 20*np.log10(np.absolute(range_compressed[:,0])),'r')
axa.set_xlim([730,750])
axa.set_ylim([-150,-60])


###ford
dxx=int((10-6.8)/0.05)
dyy=int((10-2.3)/0.05)
eks=int(8.6/0.05)

IV2=I[dxx:dxx+eks,dyy:dyy+eks]
x_vec2=np.linspace(-6.8, 1.8, num=171) #Scene center er sat i (x=0,y=0)
y_vec2=np.linspace(-2.3,6.3, num=171)

I_DBV2=20*np.log10(np.abs(IV2)/np.max(np.abs(I)))
fig, axa = plt.subplots()
axa.set_title('f')
fpf=axa.pcolormesh(x_vec, y_vec, (I_DBV2[0:I_DBV2[:,1].size-1,0:I_DBV2[1,:].size-1]),cmap='gray',vmin=0.00008)#, vmax=60000)#, vmax=60000)#,vmin=0, vmax=500)
cbar=fig.colorbar(fpf,ax=axa)
#axa.set_xlim([-20,-15])
#axa.set_ylim([15,20])
plt.xlabel("Range [m]")
plt.ylabel("Azimuth [m]")
cbar.set_label("Intensity [dB]")
axa.set_aspect('equal')