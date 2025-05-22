

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
from toptica.lasersdk.client import Client, NetworkConnection
from LABsetupDef import get_freq_data,time_table
import socket
import time

# Measuring


centerF = 680 # Center frequency in GHz
BW = 500 # Bandwidth in GHz
stepSZ = 0.02 # Step size in GHz
int_time = 30 # ms per step


def measure(centerF,BW,stepSZ,int_time):
    Fmax = centerF + BW/2 + 20
    Fmin = centerF - BW/2 - 5
    stepN = int((Fmax-Fmin)/stepSZ)
    dataMat = np.zeros([4,stepN+10])
    
    TCP_IP = '192.168.54.82'
    TCP_PORT = 1998
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((TCP_IP, TCP_PORT))
    with Client(NetworkConnection(TCP_IP)) as client: # Også en connection
        # Base functions: client.get, client.set, client.exec
        print('---Connected---')
        client.set('laser-operation:emission-global-enable', True) # Tænder laser
        # Reads and sets default parameters of antennas 
        offset=client.get('lockin:mod-out-offset-default')
        ampl=client.get('lockin:mod-out-amplitude-default')
        print('Parameters set... ','Offset:', offset, 'Amplitude:', ampl)
        client.set('lockin:mod-out-offset', offset)
        client.set('lockin:mod-out-amplitude', ampl)
        # Setup of scan parameters
        client.set('lockin:integration-time', int_time)
        client.set('frequency:scan-mode-fast', True)
        client.set('frequency:frequency-min', Fmin)
        client.set('frequency:frequency-max', Fmax)
        client.set('frequency:frequency-step', stepSZ)
        # Setup of scan timing
        ### Main Code ###
        client.set('frequency:frequency-set', Fmin)
        time.sleep(7)
        client.exec('frequency:fast-scan-clear-data')
        client.exec('frequency:fast-scan-start')
        d=client.get('frequency:fast-scan-isscanning')
        while d==True:
            #print('Scanning...')
            time.sleep(0.5)
            d=client.get('frequency:fast-scan-isscanning')
        # Data collection from DLC Pro
        dataMat[0:3,:] = get_freq_data(stepN, s)
        client.set('lockin:mod-out-offset', 0)
        client.set('lockin:mod-out-amplitude', 0)
        client.set('laser-operation:emission-global-enable', False)
    return dataMat

def dataconv(dataMat,centerF,BW)
    
    #import-----------------------------
    df=0.02
    freq_set=dataMat[0,:]
    i_ph=dataMat[2,:]
    freq_act=dataMat[1,:]
    '''
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.set_title('rå')
    ax1.plot(freq_set,i_ph)
    '''
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
    
    freq_set=np.linspace(centerF-BW/2, centerF+BW/2, int(freq_set.size))
    
    i_ph_int=np.interp(freq_set, freq_til_intp, i_ph_til_intp, left=None, right=None)
    del freq_til_intp, i_ph_til_intp, count
    
    i_ph_analy=np.array(hilbert(i_ph_int))
    '''
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.set_title('interpoleret')
    ax2.plot(freq_set,i_ph_int)
    #hilbert--------------------------------------------------------------
    
    f2 = plt.figure()
    ax2 = f2.add_subplot(111)
    ax2.set_title('hilbert envelope')
    ax2.plot(freq_set,np.absolute(i_ph_analy))
    '''
    
        
    #filterr=np.flip(filterr)
    
    
    
    #Fourier fra freq til tids domæne
    i_ph_tid=np.fft.ifft(i_ph_analy)
    #i_ph_tid=np.flip(i_ph_tid)
    
    N=i_ph_tid.size
    dt=1/(N*df)
    tid_vec=np.linspace(0, N-1,N)*dt
    distance=tid_vec*10**-9*3*10**8/2
    return distance, i_ph_tid

dataMat = measure(centerF=centerF,BW=BW,stepSZ=stepSZ,int_time=int_time)
distance, i_ph_tid = dataconv(dataMat=dataMat,centerF=centerF, BW=BW)


f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.set_title('i_ph_tid')
ax3.set_xlim(7.1,7.5)
#ax3.set_ylim(-0.000,0.02)
ax3.plot(distance,np.abs(i_ph_tid))

f3 = plt.figure()
ax3 = f3.add_subplot(111)
ax3.set_title('i_ph_tid [dB]')
ax3.set_xlim(6.8,7.5)
#ax3.set_ylim(-0.000,0.02)
ax3.plot(distance,20*np.log10(np.abs(i_ph_tid)/np.max(np.abs(i_ph_tid))))
ax3.set_ylim([-25,0])

#from playsound import playsound

# for playing note.wav file
#playsound('C:\\Users\\detri\\Desktop\\LMG THz Microscopy/\Goat Scream - Sound Effect (HD) [ ezmp3.cc ].mp3')
