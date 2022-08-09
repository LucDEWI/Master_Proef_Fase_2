import cv2
import numpy as np
import pyrealsense2 as rs
def add_gaussian_noise(X_img):


    # Gaussian distribution parameters
    mean = 0
    var = 0.1
    sigma = var ** 0.5


    gaussian = np.random.random((480, 640, 1)).astype(np.float32)
    gaussian = np.concatenate((gaussian, gaussian, gaussian), axis=2)
    gaussian_img = cv2.addWeighted(X_img, 0.75, 0.25 * gaussian, 0.25, 0, dtype= cv2.CV_8U)


    return gaussian_img

connection = rs.pipeline()
configuration = rs.config()
configuration.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
configuration.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)





# while True:
try:
    connection.start(configuration)
    align = rs.align(rs.stream.color)

    frames = connection.wait_for_frames()
    aligned = align.process(frames)

    color_frame = aligned.get_color_frame()
    depth_frame = aligned.get_depth_frame()
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    cv2.imshow('', color)
    cv2.waitKey(1000)

finally:
    connection.stop()
    color_noise = add_gaussian_noise(color)
    while True:
        cv2.imshow('kleur     ruis', color_noise)
        cv2.waitKey(1)