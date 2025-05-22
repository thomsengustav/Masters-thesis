import time
import numpy as np
#import msvcrt
import serial
import base64
import struct
import datetime as dt

''' 
# Code to communicate to arduino
def send_pos(ard)
    x = str(v0)
    arduino.write(bytes(x,  'utf-8'))
    c=True
    data = ''
    while c==True:
        # Read a line from serial
        data = arduino.readline()
        time.sleep(0.5)
        if data:  # If line is not empty
            c=False
'''
# Code to communicate to THz spectrometer
def send(sock,command1):
    command1 = command1 + '\n'
    res = bytes(command1, 'utf-8')
    sock.sendall(res)

# Vector creation for get_freq_data function
def create_vector(Number):
    n = Number/1000
    entries = np.floor(n)+1
    v = np.zeros(int(entries))
    for i in range(int(entries)):
        if (i+1)*1000<Number:
            v[i]=1000
        else:
            v[i]=Number-i*1000
    return v

# Command to retrieve data from THz spectrometer in raw format
def extract_bytes_from_signal(sock):
    received_data = b''  # Initialize an empty bytes object to store recieved data
    start_index = None  # Initialize start index as None
    
    while True:
        chunk = sock.recv(10)  # Recieve data from the socket
        received_data += chunk  # Append the received chunk to the data
        #print(received_data)
        # If start index is not found, find it
        start_index = received_data.find(b'"')
        end_index = received_data.rfind(b'"',start_index)
        if start_index != -1 and end_index != -1 and start_index != end_index:
            break
    extracted_bytes = received_data[start_index+1:end_index]
    return extracted_bytes

# Total code for extracting data from THz spectrometer and decoding it
def get_freq_data(Npoints,sock):
    data_index = [1,6,2] # 1=setFQ,6=actFQ,2=Photocurrent see commandreference p.20
    point_index = create_vector(Npoints)
    dataMat = np.zeros([int(len(data_index)),int(Npoints)+10])
    counter = 0
    for i in data_index:
        data = np.zeros([int(Npoints)+1])
        for j in range(len(point_index)):
            send(sock,'(exec \'frequency:fast-scan-get-data ' + str(i) + ' ' + str(j*1000) + ' ' + str(1000)+')')
            v = extract_bytes_from_signal(sock)
            decoded_bytes = base64.b64decode(v)

            # Calculate the number of floating-point values
            num_values = len(decoded_bytes) // 8

            # Unpack the bytes into floating-point numbers
            float_values = struct.unpack('<' + 'd' * num_values, decoded_bytes)
            array=np.asarray(float_values)
            if j==0:    
                data = array#np.append(v0,array)
            else:
                data = np.append(data,array)
        #print(v0)
        dataMat[counter,0:len(data[:])] = data
        counter += 1
    return dataMat
# Bad time tracking for lab software
def time_table(omgangtid,timetable,counter,aPoints):
    timetable[counter+1] = int(time.time())
    omgangtid[counter] = int(timetable[counter+1]-timetable[counter])
    totaltid = int(timetable[counter+1]-timetable[0])
    tidtilbage = totaltid/(counter+1)*aPoints-totaltid
    ht = np.floor(tidtilbage/3600)
    mt = np.floor((tidtilbage-3600*ht)/60)
    tt = tidtilbage - ht*3600 - mt*60
    ho = np.floor(omgangtid[counter]/3600)
    mo = np.floor((omgangtid[counter]-3600*ho)/60)
    to = omgangtid[counter] - ho*3600 - mo*60
    
    print('Scan ',counter+1,' of ', aPoints, ' is done.')
    if mo ==0:
        print('Scan duration: ', to, ' seconds')
    else:
        print('Scan duration: ', mo,' minutes and ', to, ' seconds')
    if ht==0:
        print('Time remaining: ', mt, ' minutes and ', tt, ' seconds.')
    else:
        print('Time remaining: ',ht,' hours, ', mt, ' minutes and ', tt, ' seconds.')
    return omgangtid,timetable

# File naming
def tidsfil(navn):
    date = dt.datetime.now()
    date = date.strftime('%Y%m%dh%H')
    fil = date + '_' + navn
    return fil
