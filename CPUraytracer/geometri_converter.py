from IPython import get_ipython
get_ipython().magic('reset -sf')
import matplotlib.pyplot as plt
import numpy as np
import trimesh


'''
Load .glb file and convert to np.array 
'''

# load .glb file
scene = trimesh.load("RT_speed_ref_4.glb")
geometries = list(scene.geometry.values())


scene.show()



#tæl geometrier og totale antal trekanter
nr_geometries=len(geometries)
antal_triangle_tot=0
for u in range(0,nr_geometries):
    geometry = geometries[u]
    triangles=np.array(geometry.triangles)
    nr_triangles=triangles[:,0,0].size
    antal_triangle_tot=antal_triangle_tot+nr_triangles

#sæt alle trekanter i en enkel matrice
nr_triangles2=0  
tot_triangles=np.zeros([antal_triangle_tot,3,3])
for u in range(0,nr_geometries):
    geometry = geometries[u]
    triangles2=np.array(geometry.triangles)
    tot_triangles[nr_triangles2:triangles2[:,0,0].size+nr_triangles2,:,:]=triangles2
    
    nr_triangles2=triangles2[:,0,0].size+nr_triangles2
    
plot_geo=1
if plot_geo==1:
    fig = plt.figure(figsize = (8,8))
    ax = plt.axes(projection='3d')
    ax.grid()
    for i in range(0,antal_triangle_tot):
            z=np.array([tot_triangles[i,0,2],tot_triangles[i,1,2],tot_triangles[i,2,2]])
            y=np.array([tot_triangles[i,0,1],tot_triangles[i,1,1],tot_triangles[i,2,1]])
            x=np.array([tot_triangles[i,0,0],tot_triangles[i,1,0],tot_triangles[i,2,0]])
            ax.plot3D(x, y, z)
            ax.axis('equal')


    


