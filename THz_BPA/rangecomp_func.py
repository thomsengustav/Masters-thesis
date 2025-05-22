import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.colors 
custom_map_linlog = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#000000", "#3F3F3F","#737373","#989898","#B3B3B3","#C8C8C8","#D8D8D8","#E6E6E6","#F3F3F3","#FFFFFF"])
from scipy import signal



def range_compression(i_ph_tid, c, dt, ref_puls, plot):
    range_compressed=np.zeros((i_ph_tid[:,0].size,i_ph_tid[0,:].size), dtype=np.complex_)
    for i in range(0,i_ph_tid[0,:].size):
        range_compressed[:,i]=signal.fftconvolve(i_ph_tid[:,i], np.conjugate(ref_puls),mode='same')*dt
    distance2=np.linspace(0-ref_puls.size, range_compressed[:,0].size-1-ref_puls.size,num=range_compressed[:,0].size)*dt*c/2+0.02
    if plot==1:
        f2 = plt.figure()
        ax2 = f2.add_subplot(111)
        ax2.set_title('matched')
        ax2.set_xlim(3.45,3.49)
        ax2.plot(distance2,np.absolute((range_compressed[:,70])))
        xv=np.linspace(0,i_ph_tid[0,:].size-1,num=i_ph_tid[0,:].size)

        fig, axa = plt.subplots()
        axa.set_title('range compressed  profile amplitude')
        axa.set_xlim(3.35,3.5)
        fp3=axa.pcolormesh(distance2, xv, np.transpose(np.absolute(range_compressed[0:range_compressed[:,1].size-1,0:range_compressed[1,:].size-1])), cmap=custom_map_linlog)
        cbar=fig.colorbar(fp3,ax=axa)
        plt.xlabel("Range (m)")
        plt.ylabel("Azimuth step")
        cbar.set_label("Intensity (a.u)")
        fig, axa = plt.subplots()
        axa.set_title('range compressed  profile phase')
        axa.set_xlim(3.4,3.5)
        fp3=axa.pcolormesh(distance2, xv, np.transpose(np.angle(range_compressed[0:range_compressed[:,1].size-1,0:range_compressed[1,:].size-1])))
        cbar=fig.colorbar(fp3,ax=axa)
        plt.xlabel("Range (m)")
        plt.ylabel("Azimuth step")
        cbar.set_label("Intensity (a.u)")
        
    
    return range_compressed#, f2, fp3 distance2,