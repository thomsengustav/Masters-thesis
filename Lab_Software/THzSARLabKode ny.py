
# Importing relevant packages
import time; import numpy as np; import socket; import pyvisa; import os; import serial;
from toptica.lasersdk.client import Client, NetworkConnection
from LABsetupDef import get_freq_data,time_table,tidsfil

######################################
### Scan setup ###
#posmin=0; posmax=4000 # Position in steps 300 steps/degree (max degrees: 14)
aI = 10 # The angle interval we want to examine
aPoints = aI*5 # Amount of points the interval is devided into
centerF = 550 # Center frequency in GHz
BW = 300 # Bandwidth in GHz
stepSZ = 0.04 # Step size in GHz
int_time = 50 # ms per step
Name = 'Corner_ref'
dH = 0 #38 #np.array([4,0])
rot = True

Inf = """Information om prøven...
Beskrivelse: corner reference
Range: 80 cm
Slant: app. 10 deg
Integration angle: {aI} deg
Center frequency: {centerF} GHz
Bandwidth: {BW} GHz
Integration Time: {int_time} ms
Frequency Step Size: {stepSZ} GHz
Polarisering: VV
"""
######################################

filename = tidsfil(Name)

#### This part shouldn't be modified ####
#Python setup
counter = 0
v = np.linspace(0,aI,aPoints)
#v = np.floor(v)
Fmax = centerF + BW/2 + 20
Fmin = centerF - BW/2 - 5
stepN = int((Fmax-Fmin)/stepSZ)
dataMat = np.zeros([4,stepN+10,int(len(v))])
omgangtid = np.zeros(aPoints)

rm = pyvisa.ResourceManager()


if rot == True:
    ns = rm.open_resource('GPIB0::6::INSTR')
    # Initialize Nanostepper
    ns.query('RO0=1')
    ns.query('V=0.3')
    # Ratio of conversion
    gpi = 218.5/30
    ns.query('MR0=' + str((aI)/(2*gpi)))
    time.sleep(60)
if dH != 0:
    LCR = 50/33
    ns.query('V1=5'); ns.query('RO1=0'); ns.query('CAL1=+')
    ns.query('MCAL1')
    time.sleep(60)
    ns.query('MR1=-'+str(3+dH*LCR))
    time.sleep(100)

TCP_IP = '192.168.54.82'
TCP_PORT = 1998
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect((TCP_IP, TCP_PORT))
    

### Main Setup ###
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
    timetable = np.zeros(aPoints+10)
    timetable[0] = time.time()
    ### Main Code ###
    for v0 in v:
        client.set('frequency:frequency-set', Fmin)
        time.sleep(7)
        client.exec('frequency:fast-scan-clear-data')
        client.exec('frequency:fast-scan-start')
        d=client.get('frequency:fast-scan-isscanning')
        print('Position:',v0,' Scanning...')
        while d==True:
            #print('Scanning...')
            time.sleep(0.5)
            d=client.get('frequency:fast-scan-isscanning')
        # Data collection from DLC Pro
        dataMat[0:3,:,counter] = get_freq_data(stepN, s)
        dataMat[3,0,counter] = v0; dataMat[3,1,counter] = dH
        # Timing
        omgangtid,timetable = time_table(omgangtid,timetable,counter,aPoints)
        if rot == True:
            ns.query('MR0=-' + str(v[1]/gpi))
        print('-------------------')
        counter += 1
    time.sleep(3)
    client.set('lockin:mod-out-offset', 0)
    client.set('lockin:mod-out-amplitude', 0)
    client.set('laser-operation:emission-global-enable', False)
    np.save(os.path.join('data', filename+'.npy'),dataMat)
    #Write information file #
    with open('C:\\Users\\detri\Desktop\\LMG THz Microscopy\\sampleinf\\' + filename + '.txt','w') as file:
        file.write(Inf)
if rot == True:
    ns.query('MR0=' + str((aI)/(2*gpi)))
time.sleep(60)
if dH != 0:
    ns.query('MCAL1')
    time.sleep(30)
if rot == True:
    ns.close()
print('Scan finished')