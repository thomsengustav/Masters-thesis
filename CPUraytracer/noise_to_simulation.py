from IPython import get_ipython
get_ipython().magic('reset -sf')
import numpy as np
# import cupy as cp
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib.colors 
import time
custom_map_linlog = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#000000", "#3F3F3F","#737373","#989898","#B3B3B3","#C8C8C8","#D8D8D8","#E6E6E6","#F3F3F3","#FFFFFF"])


'''
Add signal noise to simulations.
The script contains the total pipeline from simulation histogram to SAR image
Along different steps of the process noise can be added to the signal
'''


plot=0

# load histogram
rangeV=np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\yacht_lang_clut.txt")
rangeVV = np.loadtxt("C:\\Users\\thoms\\.spyder-py3\\Optimering af ray tracer\\yacht_lang.txt")

max1 = np.max(rangeV)
max2 = np.max(rangeVV)

# set rangeV
rangeV = rangeV/max1*2 + rangeVV/max2
rangeV = rangeVV

rangeV=rangeV.astype(np.complex128)
binsize=0.016
antalmål= 900 # 240
indgange=80000
rangeV=rangeV[0:antalmål*indgange]
nr=np.linspace(0,antalmål-1,antalmål)

radius=90*1.4
slant=10*np.pi/180
radar_range=250*2
vinkelint= 209 # 40


hist_conv = False
noise_sig = False

int_plot = False
dB_plot = True
im_noise_plot = False


addvar_val = 0.01
mulvar_val = 2
var_azi = 0 # 0.2


range_vec=np.linspace(0, indgange*binsize, indgange)

f52 = plt.figure()
ax52 = f52.add_subplot(111)
ax52.set_title('rangeV')
ax52.plot(rangeV[0:indgange])
rangeV2=np.zeros(rangeV.size, dtype=np.complex128)

rangeV2=0.5*rangeV

ff=np.isnan(rangeV2)
fff=np.asarray(np.where(ff==1))
ffi=np.isinf(rangeV2)
fffi=np.asarray(np.where(ffi==1))
print(fffi.sum())
print(ff.sum())

c = 3e8
T = 1280 / c

df = 1/T

f_vec = np.zeros(indgange)
for i in range(indgange):
    f_vec[i] = i*df / 1e9

# a=fff[0,0]

# print(rangeV2[fff[0,0]])
# print(rangeV[1,0])
# print(rangeV[a-1,1])
# print(rangeV[a-1,2])

print(rangeV[fff[0]])
rangeV2[fff[0]]=0;
rangeV2[fffi[0]]=0;

# f52 = plt.figure()
# ax52 = f52.add_subplot(111)
# ax52.set_title('rangeV2')
# ax52.plot(np.abs(rangeV2[0:indgange]))

rangeM=rangeV2.reshape((antalmål,indgange))

del rangeV
del rangeV2

f5 = plt.figure()
ax5 = f5.add_subplot(111)
ax5.set_title('Range plot')
ax5.set_ylabel('Intensity')
ax5.set_xlabel('range [m]')
ax5.plot(range_vec,rangeM[0,:])
ax5.set_xlim([1080,1180])



def renorm_fkt(vec):
    max_val = np.max(vec)
    # use sqrt(x) for renormalization
    vec_unit = vec / max_val
    new_vec = np.sqrt(vec_unit)
    new_vec = new_vec * max_val
    return new_vec

##################### BLUR BABY BLUR ##############################

# perform blur of the histogram
xGauss=np.linspace(-2,2, 30)
Gauss_sig=1
Gauss=1/(Gauss_sig*np.sqrt(2*np.pi))*np.exp(-xGauss**2/(2*Gauss_sig**2))
f5 = plt.figure()
ax5 = f5.add_subplot(111)
ax5.plot(Gauss)
blur_vec = np.ones(1)


X_multi=np.zeros((rangeM[:,0].size,rangeM[0,:].size), dtype=np.complex128)
for j in range(0,rangeM[:,0].size):
    N = np.random.normal(250, 30, 1)
    index_max = np.where(rangeM[j,:] == np.max(np.real(rangeM[j,:])))[0]
    
    X_multi[j,:] = signal.fftconvolve(np.real(rangeM[j,:]), Gauss,'same') #+ blur_val/ np.max(blur_val)*10000

blur_val = signal.fftconvolve(np.real(rangeM[j,:]), np.ones(20000),'same')




# f5 = plt.figure()
# ax5 = f5.add_subplot(111)
# ax5.set_title('Multi')
# ax5.plot(range_vec,blur_val)
# ax5.set_xlim([0,1300])

f5 = plt.figure()
ax5 = f5.add_subplot(111)
ax5.set_title('Multi')
ax5.plot(range_vec,X_multi[0,:])
ax5.set_xlim([1100,1125])

fft_range = np.fft.fft(X_multi[0,:])




##### use convolution!
if hist_conv == True:
    rangeM=X_multi





##########################################3

rangeM2=rangeM[:,66000:76000]
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
f0=0.34*10**9
B_c=0.25*10**9
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

t1_vec=np.linspace(0, dt*(indgange-1), num=indgange)
modulator=np.zeros(t1_vec.size, dtype=np.complex_)
for i in range(0,t1_vec.size):
    modulator[i]=np.exp(-1j*2*np.pi*f0*t1_vec[i])



######################################################################
rangeM_signal=np.zeros((antalmål, indgange), dtype=np.complex128)
rangeM_signal_no_mod=np.zeros((antalmål, indgange), dtype=np.complex128)
smooth_sig = np.zeros((antalmål, indgange), dtype=np.complex128)
# fft_sig = np.zeros((antalmål, indgange), dtype=np.complex128)
xGauss=np.linspace(-2,2, 30)
Gauss_sig=2
Gauss=1/(Gauss_sig*np.sqrt(2*np.pi))*np.exp(-xGauss**2/(2*Gauss_sig**2))

# calculate signals
for i in range(0,antalmål):
    rangeM_signal[i,:]=(signal.fftconvolve(rangeM[i,:], enkel_puls,'same'))*modulator
    rangeM_signal_no_mod[i,:]=(signal.fftconvolve(rangeM[i,:], enkel_puls,'same'))
    # smooth_sig[i,:] = signal.fftconvolve(rangeM_signal_no_mod[i,:], Gauss,'same')
    # fft_sig[i,:] = np.fft.fft(rangeM_signal_no_mod[i,:])
    

# noise up that signal babay!
new_sig = np.zeros((antalmål,indgange), dtype=np.complex128)

# setup filter

start = 200
filter_var = np.ones(indgange)
filter_x = np.linspace(0,indgange,indgange)

gauss_filter_x = np.linspace(0,indgange,indgange-start)
Gauss_sig = 400
gauss_filter = 1/(Gauss_sig*np.sqrt(2*np.pi))*np.exp(-gauss_filter_x**2/(2*Gauss_sig**2))
gauss_filter = gauss_filter / np.max(gauss_filter)
filter_var[start:indgange] = gauss_filter


filter_var2 = np.ones(indgange)
start2 = indgange - 200
gauss_filter_x2 = np.linspace(0,indgange,indgange-start2)
Gauss_sig2 = 400
gauss_filter2 = 1/(Gauss_sig2*np.sqrt(2*np.pi))*np.exp(-gauss_filter_x2**2/(2*Gauss_sig2**2))
gauss_filter2 = gauss_filter2 / np.max(gauss_filter2)
filter_var2[start2:indgange] = gauss_filter2

filter_var2 = abs(1-filter_var2)

filter_new = filter_var + filter_var2 

max_range_val = np.real(np.max(rangeM))

fAddNoise = indgange #6000
fMulNoise = indgange #6000

noiseAddVar = addvar_val * max_range_val
noiseMulVar = mulvar_val

for i in range(antalmål):
    n_fast = np.random.normal(0,noiseAddVar,fAddNoise)
    test_vec = np.fft.ifft(rangeM_signal[i,:]) 
    test_vec[0:fMulNoise] = test_vec[0:fMulNoise] *np.random.normal(1,noiseMulVar,fMulNoise)
    test_vec[0:fAddNoise] = test_vec[0:fAddNoise] + n_fast
    test_vec = test_vec * filter_new
    new_sig[i,:] = np.fft.fft(test_vec)*np.random.normal(1,var_azi,1)

plot_filter_sig = True
if plot_filter_sig == True:
    test = np.real(np.fft.fft(rangeM_signal[0,:]))
    test2 = np.imag(np.fft.fft(rangeM_signal[0,:]))
    test_max = np.max(test)
    fig, ax = plt.subplots(figsize=(4, 4))
    plt1, = ax.plot(f_vec, test/test_max, label = '$\mathcal{Re}(s_D(f))$')
    plt2, = ax.plot(f_vec, filter_new, label = 'Filter')
    ax.set_xlim([0,1.2])
    ax.set_xlabel('Frequency [GHz]', fontsize = 14)
    ax.set_ylabel('Normalized intenisty [a.u.]', fontsize = 14)
    ax.legend(handles=[plt1, plt2], fontsize = 14)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    plt1, = ax.plot(f_vec, test/test_max, label = '$\mathcal{Re}(s_D(f))$')
    #plt2, = ax.plot(f_vec, filter_new, label = 'Filter')
    ax.set_xlim([5,10])
    ax.set_xlabel('Frequency [GHz]', fontsize = 14)
    ax.set_ylabel('Normalized intenisty [a.u.]', fontsize = 14)
    #ax.legend(handles=[plt1, plt2], fontsize = 14)
    plt.show()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    plt1, = ax.plot(f_vec, test, label = '$\mathcal{Re}(s_D(f))$')
    plt2, = ax.plot(f_vec, test, label = 'Filter')
    ax.set_xlim([0,0.15])
    ax.set_xlabel('Frequency [GHz]', fontsize = 14)
    ax.set_ylabel('Normalized intenisty [a.u.]', fontsize = 14)
    ax.legend(handles=[plt1, plt2], fontsize = 14)
    plt.show()
    
    # plt.plot(range_vec, np.real(rangeM_signal[0,:]))
    # plt.xlim([1080,1180])
    # plt.show()
    
    # plt.plot(range_vec, np.real(new_sig[0,:]))
    # plt.xlim([1080,1180])
    # plt.show()
    
    fig, ax = plt.subplots(figsize=(4, 4))
    plt1, = ax.plot(range_vec, np.real(rangeM_signal[0,:]), label = 'Original Signal')
    plt2, = ax.plot(range_vec, np.real(new_sig[0,:]), label = 'Filtered Signal')
    ax.set_xlim([1080,1180])
    ax.set_xlabel('Range [m]', fontsize = 14)
    ax.set_ylabel('Intensity [a.u.]', fontsize = 14)
    ax.legend(handles=[plt1, plt2], fontsize = 14)
    plt.show()



### used noise sig!
if noise_sig == True:
    rangeM_signal = new_sig


if plot==1:
    fig, axa4 = plt.subplots()
    axa4.set_title('rangeM signal')
    axa4.pcolormesh(range_vec,nr,np.abs(rangeM_signal))
    #axa4.set_xlim([60,320]) 
 
#########################nedenfor ligesom RT BPA#############################

range_prof=rangeM_signal
azi_v=np.linspace(0,vinkelint*(antalmål-1)/antalmål,antalmål)*np.pi/180-vinkelint/2*np.pi/180+np.pi/2 +180*np.pi/180


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
# f3= plt.figure()
# ax3 = f3.add_subplot(111)
# ax3.set_title('range_ref')
# ax3.plot(tid_enkel,range_ref)



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
    fig, axa = plt.subplots()
    axa.set_title('range compressed amplitude')
    axa.plot(np.absolute(range_compressed2[0,:]))

#Til billede--------------------------------------------------------------------------
x_vec=np.linspace(-30, 30, num=300) #Scene center er sat i (x=0,y=0)
y_vec=np.linspace(-15, 15, num=300) #x og y interval i meter, num=antal pixels i x og y dim
I_puls=np.zeros([azi_v.size], dtype=np.complex_)
I=np.zeros([x_vec.size,y_vec.size], dtype=np.complex_)#matrice med pixel værdier

#position af emitter og detector, her er det sat til monostatic
pos_emitter=np.transpose(R_Mat)
pos_detector=np.zeros((3,antalmål))
for i in range(0,antalmål):
    pos_detector[:,i]=pos_emitter[:,i]+pos_emitter[:,i]/np.linalg.norm(pos_emitter[:,i])*1.1*radius
    


range_compressed=np.transpose(range_compressed)
range_compressed=np.flip(range_compressed, axis=1)

filterr=0

# fig, axa = plt.subplots()
# axa.set_title('range compressed amplitude')
# axa.plot(distance3,np.absolute(range_compressed[:,0]))

if filterr==1:
    df=1/(indgange*dt)
    bw_ingange=int(B_c/df/2)
    window = signal.windows.taylor(bw_ingange*2, nbar=40, sll=35, norm=False)
    
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
nul1=0

#Laver et billede, cupy skal være hentet, ellers skal Billede_danner lige rettes
opdel_pulser=30

tidtager=time.time()
from Billede_danner_GPU_bistatic import Billede_danner_GPU_bistatic


for i in range(0,opdel_pulser):
    index1=int(i*antalmål/opdel_pulser)
    index2=int((i+1)*antalmål/opdel_pulser)
    I+=Billede_danner_GPU_bistatic(y_vec,x_vec, pos_emitter[:,index1:index2], pos_detector[:,index1:index2], nul1, d_dist, k, range_compressed[:,index1:index2])
    # print(i)

print(time.time()-tidtager)




def get_i_shift(I, x_shift, y_shift):
    I_shift = np.zeros((300,300), dtype = np.complex128)
    I_shift[x_shift:, y_shift:] = I[0:-x_shift, 0:-y_shift]
    return I_shift


if int_plot == True:
    fig, axa = plt.subplots()
    axa.set_title('image')
    fpf=axa.pcolormesh(y_vec, x_vec, np.absolute(I[0:I[:,1].size-1,0:I[1,:].size-1]), cmap=custom_map_linlog)#, vmax=0.3)
    cbar=fig.colorbar(fpf,ax=axa)
    plt.xlabel("Range(x) (m)")
    plt.ylabel("Azimuth(y) (m)")
    cbar.set_label("Intensity (a.u.)")
    axa.set_aspect('equal')


I_DB=20*np.log10(np.abs(I)/np.max(np.abs(I)))
if dB_plot == True:
    fig, axa = plt.subplots()
    #axa.set_title('Sim w. noise')
    fpf=axa.pcolormesh(y_vec, x_vec, (I_DB[0:I_DB[:,1].size-1,0:I_DB[1,:].size-1]),cmap='gray',vmin=-20)#, vmax=60000)#, vmax=60000)#,vmin=0, vmax=500)
    cbar=fig.colorbar(fpf,ax=axa)
    #axa.set_xlim([-20,-15])
    # axa.set_ylim([15,20])
    plt.xlabel("Range [m]", fontsize = 16)
    plt.ylabel("Azimuth [m]", fontsize = 16)
    cbar.set_label("Intensity [dB]", fontsize = 14)
    axa.set_aspect('equal')



# image blur
gauss_f_2d = np.zeros((100,100))
x_gvec0 = np.linspace(1,300,100)
y_gvec0 = np.linspace(1,300,100)

X, Y = np.meshgrid(x_gvec0, y_gvec0)

for i in range(100):
    gauss_f_2d[i,:] = np.random.normal(1,0.3,100)


x_gvec = np.linspace(1,300,300)
y_gvec = np.linspace(1,300,300)

interp_noise = np.repeat(gauss_f_2d, 3, axis=1).repeat(3, axis=0)

# plt.pcolor(gauss_f_2d, cmap='gray')
# plt.show()


I_noise = np.zeros((300,300), dtype= np.complex128)
for i in range(300):
    I_noise[i,:] = I[i,:]*interp_noise[i,:]

I_DB_n=20*np.log10(np.abs(I_noise)/np.max(np.abs(I_noise)))

if im_noise_plot == True:
    fig, axa = plt.subplots()
    axa.set_title('Image noise')
    fpf=axa.pcolormesh(y_vec, x_vec, (I_DB_n[0:I_DB_n[:,1].size-1,0:I_DB_n[1,:].size-1]),cmap='gray',vmin=-20)#, vmax=60000)#, vmax=60000)#,vmin=0, vmax=500)
    cbar=fig.colorbar(fpf,ax=axa)
    #axa.set_xlim([-20,-15])
    # axa.set_ylim([15,20])
    plt.xlabel("Range [m]", fontsize = 16)
    plt.ylabel("Azimuth [m]", fontsize = 16)
    cbar.set_label("Intensity [dB]", fontsize = 14)
    axa.set_aspect('equal')

