import numpy as np
import os
import cv2


#%%
def write_obj(outpath, T, H, h_scale=3):
    if H.dtype=='uint8':
        H = H.astype('float') / 35.3516
    elif H.dtype=='uint16':
        H = H.astype('float') / 9052.32
    if len(H.shape)==3:
        H = H[:,:,0]
    H = H*h_scale
    H = H[::-1,:]
    
    V = []
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            V.append([i, j, H[i,j]])
    
    F = []
    for i in range(H.shape[0]-1):
        for j in range(H.shape[1]-1):
            F.append([i*H.shape[1]+j, i*H.shape[1]+j+1, (i+1)*H.shape[1]+j])
            F.append([(i+1)*H.shape[1]+j, i*H.shape[1]+j+1, (i+1)*H.shape[1]+j+1])
    
    with open(os.path.join(outpath, 'Map.obj'), 'w') as file:
        file.write('mtllib Map.mtl\n')
        file.write('usemtl Map\n')
        file.write('\to Map\n')
        for v in V:
            file.write("\t\tv %.4f %.4f %.4f\n" % (v[0], v[2], v[1]))
        for v in V:
            file.write("\t\tvt %.4f %.4f\n" % (v[1]/H.shape[1], v[0]/H.shape[0]))
        file.write("\t\ts 1")
        for f in F:
            file.write("\t\t\tf %d/%d %d/%d %d/%d\n" % (f[0]+1, f[0]+1, f[1]+1, f[1]+1, f[2]+1, f[2]+1))
    
    with open(os.path.join(outpath, 'Map.mtl'), 'w') as file:
        file.write("newmtl Map\nKa 1.000 1.000 1.000\nKd 1.000 1.000 1.000\nKs 0.000 0.000 0.000\nillum 1\nmap_Ka MapTexture.png\nmap_Kd MapTexture.png")
    
    cv2.imwrite(os.path.join(outpath,'MapTexture.png'), T.astype('uint8'))