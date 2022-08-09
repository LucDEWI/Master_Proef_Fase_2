#bibliotheken
import cv2
import numpy as np
import cv2 as cv
import os
from src import Functions as func

#Geeft een visualisatie van de locatie van het referentiestelsel. Uitendelijke kalibratie gebeurt via het EXTR_cal script

#eigenschappen patroon
h=14
b=9
size=17.6 #aantal mm zijde van een vierkant op het schaakbord

#laden van intrinsieke gegevens uit data
datapath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'
mtx = np.load(datapath+'\intrinsics.npy')
dist = np.load(datapath+'\distortion.npy')

#venster instellingen
window_name = 'Extrinsic calibration'
window_b, window_h = 640, 480

#camera instellingen
color_resolution = (640, 480)
depth_resolution = (640, 480)
frames_per_second = 30

#stel venster en camera in
func.setwindow(window_name, window_b, window_h)
pipeline, align_object = func.start(color_resolution, depth_resolution, frames_per_second)

#functie die niets doet, nodig voor sliders
def nothing(*args):
    pass



def draw(img, corners, imgpts):
    corners = corners.astype('int32')
    imgpts = imgpts.astype('int32')
    corner = tuple(corners[0].ravel())

    img = cv.line(img, corner, tuple(imgpts[0].ravel()), (255,0,0), 5)
    img = cv.line(img, corner, tuple(imgpts[1].ravel()), (0,255,0), 5)
    img = cv.line(img, corner, tuple(imgpts[2].ravel()), (0,0,255), 5)
    return img

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((b*h,3), np.float32)
objp[:,:2] = np.mgrid[0:b,0:h].T.reshape(-1,2)
objp = 17.6*objp
axis = np.float32([[60,0,0], [0,60,0], [0,0,-60]]).reshape(-1,3)

while True:

    img, depth_img = func.getframes(pipeline, align_object)
    img = img[0:400, 100:500]
    img = cv2.resize(img, (300, 300), interpolation= cv2.INTER_AREA)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ret, corners = cv.findChessboardCornersSB(img, (b, h), cv.CALIB_CB_MARKER)

    if ret == True:

        ret, rvecs, tvecs, inliers = cv.solvePnPRansac(objp, corners, mtx, dist)

        imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

        beeld = draw(img, corners, imgpts)
        cv.imshow('Extrinsic calibration', beeld)
        cv.waitKey(1)

    else:
        cv.imshow('Extrinsic calibration', img)
        cv.waitKey(1)

    if  cv.waitKey(1) == ord('q'):
        break


cv.destroyAllWindows()



