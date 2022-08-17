import glob
import os
import numpy as np
from imageio import imsave
import argparse
import cv2

from utils.dataset_processing.image import DepthImage

# dit scrpit voegt ruis toe aan de cornell dataset
# neem een kopie van de originele dataset, want het is niet mogelijk het origineel tergu te krijgen

# functie die gaussiaanse ruis toevoegt aan de dataset
def noise(img):

    mean = 0
    var = 0.001
    sigma = var ** 0.5
    gaussian = (np.random.normal(mean, sigma, (480, 640)))#  np.zeros((224, 224), np.float32)

    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian



    # cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.float32)
    return noisy_image




if __name__ == '__main__':
    # locatie van de dataset
    parser = argparse.ArgumentParser(description='Generate depth images from Cornell PCD files.')
    parser.add_argument('path', type=str, help='Path to Cornell Grasping Dataset')
    args = 'C:\cornell_4'

    lis = glob.glob(os.path.join(args, '*', 'pcd*[0-9]*d.tiff'))
    lis.sort()
    # toevoegen van ruis
    for li in lis:
        di = DepthImage.from_tiff(li)
        di = noise(di)

        print(li)
        imsave(li, di.astype(np.float32))