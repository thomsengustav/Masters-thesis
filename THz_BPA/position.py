# from IPython import get_ipython
# get_ipython().magic('reset -sf')
# import matplotlib.pyplot as plt
import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib as mpl
import matplotlib.colors 
custom_map_linlog = matplotlib.colors.LinearSegmentedColormap.from_list("custom", ["#000000", "#3F3F3F","#737373","#989898","#B3B3B3","#C8C8C8","#D8D8D8","#E6E6E6","#F3F3F3","#FFFFFF"])

# from matplotlib import cm
# from celluloid import Camera

def Positionsregner(d_em, d_dec, phi_v, d_bistatic_angle, theta):
    d_em_plan=np.cos(theta)*d_em
    d_dec_plan=np.cos(theta)*d_dec

    pos_emitter=np.zeros((3,phi_v.size))
    pos_detector=np.zeros((3,phi_v.size))

    pos_em_xzero=-d_em_plan*np.cos(d_bistatic_angle/2)
    pos_em_yzero=-d_em_plan*np.sin(d_bistatic_angle/2)

    pos_dec_xzero=-d_dec_plan*np.cos(d_bistatic_angle/2)
    pos_dec_yzero=d_dec_plan*np.sin(d_bistatic_angle/2)

    pos_emitter_zero=np.array([[pos_em_xzero],[pos_em_yzero]])
    pos_detector_zero=np.array([[pos_dec_xzero],[pos_dec_yzero]])
    for i in range(0,phi_v.size):
        rotation=np.array([[np.cos(phi_v[i]), -np.sin(phi_v[i])],[np.sin(phi_v[i]), np.cos(phi_v[i])]])
        pos_emitteri=rotation@pos_emitter_zero
        pos_detectori=rotation@pos_detector_zero
        
        pos_emitter[0:2,i]=pos_emitteri[:,0]
        pos_detector[0:2,i]=pos_detectori[:,0]
    pos_emitter[2,:]=np.sin(theta)*d_em
    pos_detector[2,:]=np.sin(theta)*d_dec


    # camera = Camera(plt.figure())
    # for i in range(phi_v.size):
        
    #     plt.scatter(pos_emitter[1,i], pos_emitter[0,i])
    #     plt.scatter(pos_detector[1,i], pos_detector[0,i])
    #     plt.plot([0,pos_emitter[1,i]],[0,pos_emitter[0,i]])
    #     plt.plot([0,pos_detector[1,i]],[0,pos_detector[0,i]])
    #     ax = plt.gca()
    #     ax.set_aspect('equal', adjustable='box')
    #     camera.snap()
           
    # anim = camera.animate(blit=True)
    # #anim.save('scatter.mp4')
    
    return pos_emitter, pos_detector#, anim
