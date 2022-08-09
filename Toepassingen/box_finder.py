import pyrealsense2 as rs
import numpy as np
import os
import cv2



#resolutie en fps van de beelden
cres = (1920, 1080)
dres = (1280, 720)
fps = 30

datapath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'

try:
    cropped_new = np.load(datapath+'\e_box.npy')
except FileNotFoundError:
    cropped_new = np.save(datapath+'\e_box.npy', [0, 0, 0, 0])


#connectie starten
connection = rs.pipeline()

configuration = rs.config()
configuration.enable_stream(rs.stream.color, cres[0], cres[1], rs.format.bgr8, 30)
configuration.enable_stream(rs.stream.depth, dres[0], dres[1], rs.format.z16, 30)

connection.start(configuration)

align = rs.align(rs.stream.color)


while True:
    frames = connection.wait_for_frames()

    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())

    depth_color = cv2.applyColorMap(cv2.convertScaleAbs(depth, alpha=0.03), cv2.COLORMAP_JET)
    img = np.hstack((color, depth_color))
    #cv2.imshow('f', img)
    blur = cv2.GaussianBlur(depth, (3, 3), 0)
    mask = cv2.inRange(blur, 500, 530)

    mask = cv2.erode(mask, None, iterations= 2)
    mask = cv2.dilate(mask, None, iterations= 2)

    thresh_c = cv2.bitwise_and(depth_color, depth_color, mask= mask)
    canny = cv2.Canny(thresh_c, 100, 200)
    cv2.imshow('', thresh_c)
    contours, _ = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key= cv2.contourArea, reverse= True)




    #((x, y), (width, height), a) = rect =  cv2.minAreaRect(contours[0])
    x, y, w, h = box = cv2.boundingRect(contours[0])

    #box = cv2.boxPoints(rect)
    #box = np.int0(box)

    cv2.rectangle(color, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #cv2.drawContours(black, [box], 0, (255, 255, 255), -1 )
    #cv2.drawContours(color, [box], 0, (0, 0, 255), 2)
    #cv2.drawContours(color, contours, -1, (255, 0, 0))
    #final = cv2.bitwise_and(color, color, mask = np.squeeze(black))
    #final_depth = cv2.bitwise_and(depth, depth, mask= black)
    #print (width)
    #print(height)

    #cv2.imshow('', color)
    #cropped_image = color[y:y + h, x:x + w]
    print([x, y, w, h])
    x_new = x+4
    y_new = y+4
    w_new = w-4
    h_new = h-4
    cropped_new = np.array([x_new, y_new, w_new, h_new])
    cropped_new_image = color[y_new: y_new+h_new, x_new:x_new+w_new]
    #cv2.imshow('',color)
    cv2.imshow('', cropped_new_image )



    if cv2.waitKey(2000) == ord('q'):
        break

cv2.destroyAllWindows()
np.save(datapath + '\e_box.npy', cropped_new )
print('saved')