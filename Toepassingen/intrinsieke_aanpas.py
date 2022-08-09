import os
import numpy as np



datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'



#intrinsieke en extrinsieke gegevens importeren
mtx=np.load(datapath+'\intrinsics.npy')
dist=np.load(datapath+'\distortion.npy')
ext=np.load(datapath+'\extrinsics.npy')
rvecs=np.load(datapath+'\extrinsic_rvecs.npy')
tvecs=np.load(datapath+'\extrinsic_tvecs.npy')

mtx[0][2] = 960
mtx[1][2] = 540

print(mtx[0][2])
print(mtx[1][2])
print(mtx)

np.save(datapath+'\intrinsics.npy',mtx)