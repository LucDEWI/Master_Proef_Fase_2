import imageio
import cv2
import numpy as np

# geeft een beeld uit de dataset
img = imageio.imread(r'C:\cornell_4\01\pcd0104d.tiff')
while True:
    depth_scale = np.abs(img).max()
    depth_crop = img.astype(np.float32) / depth_scale

    cv2.imshow('', img)
    cv2.waitKey(1)
