import numpy as np
import cupy as cp


#cp. kan erstattes med np. hvis cupy er træls eller grafikkortet løber tør for ram
def Billede_danner_GPU_bistatic_V2(y_vec,x_vec, pos_emitter, pos_detector, nul1, d_dist, k, range_compressed):
    yv,xv=np.meshgrid(y_vec,x_vec)
    xv=np.reshape(xv,xv.size)
    yv=np.reshape(yv,yv.size)
    xv2=np.tile(xv,pos_emitter[0,:].size).reshape(xv.size, pos_emitter[0,:].size, order='F')
    yv2=np.tile(yv,pos_emitter[0,:].size).reshape(yv.size, pos_emitter[0,:].size, order='F')

    
    
    GPU_pos_emitter=cp.asarray(pos_emitter)
    GPU_pos_detector=cp.asarray(pos_detector)

    
    GPU_ref_scat=cp.zeros((xv.size,3,pos_emitter[0,:].size))
    GPU_ref_scat[:,0,:]=cp.asarray(xv2)
    GPU_ref_scat[:,1,:]=cp.asarray(yv2)

    GPU_nul1_og_d_dist=cp.asarray([nul1,d_dist,k*2])
    GPU_element=cp.zeros((xv.size,pos_emitter[0,:].size))

    GPU_element_corr2=cp.linspace(0, GPU_element[0,:].size-1, num=GPU_element[0,:].size, dtype=np.int32)
    A_mat_shap=pos_emitter[0,:].size

    GPU_element_corr2=cp.tile(GPU_element_corr2,GPU_element[:,0].size).reshape(GPU_element[:,0].size,GPU_element_corr2.size)
        
    GPU_diff1=(GPU_ref_scat-GPU_pos_emitter)#+(GPU_ref_scat-GPU_pos_detector)
    GPU_diff2=(GPU_ref_scat-GPU_pos_detector)
    cp.cuda.stream.get_current_stream().synchronize()
    
    GPU_dn=cp.linalg.norm(GPU_diff2, axis=1)
    GPU_dn+=cp.linalg.norm(GPU_diff1, axis=1)
    
    del (GPU_diff1)
    del (GPU_diff2)
    GPU_dn=GPU_dn/2
    
    GPU_element=GPU_dn/GPU_nul1_og_d_dist[1]+GPU_nul1_og_d_dist[0]
    GPU_element=GPU_element.astype(cp.int32)

    GPU_element=GPU_element*A_mat_shap+GPU_element_corr2
    

    GPU_range_compressed=cp.asarray(range_compressed)
    GPU_I_puls=cp.take(GPU_range_compressed,GPU_element)
    
    del (GPU_range_compressed)


    GPU_azi_v=1j*GPU_nul1_og_d_dist[2]*GPU_dn
    
    GPU_azi=cp.exp(GPU_azi_v)
    

    GPU_I_puls=cp.multiply(GPU_I_puls,GPU_azi)
    
    GPU_I_vec=cp.sum(GPU_I_puls,axis=1)
    GPU_I=GPU_I_vec.reshape(x_vec.size,y_vec.size)
    

    I=cp.asnumpy(GPU_I)
    I_puls=cp.asnumpy(GPU_I_puls)
    azi_v=cp.asnumpy(GPU_azi_v)#.astype(np.complex64)
    #print(time.time()-tidtager)
    return I, I_puls, azi_v



