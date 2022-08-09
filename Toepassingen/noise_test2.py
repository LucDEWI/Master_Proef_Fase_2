import numpy as np
import cv2
import pyrealsense2 as rs


def noise(img):

    mean = 0
    var = 0.00001
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (480, 640)) #  np.zeros((224, 224), np.float32)
    print(gaussian.max())
    noisy_image = np.zeros(img.shape, np.float32)

    if len(img.shape) == 2:
        noisy_image = img + gaussian
    else:
        noisy_image[:, :, 0] = img[:, :, 0] + gaussian
        noisy_image[:, :, 1] = img[:, :, 1] + gaussian
        noisy_image[:, :, 2] = img[:, :, 2] + gaussian


    cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
    noisy_image = noisy_image.astype(np.uint8)
    return noisy_image
connection = rs.pipeline()
configuration = rs.config()
configuration.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
configuration.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

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
    color_noise = noise(color)
    while True:
        cv2.imshow('kleur     ruis', color_noise)
        cv2.waitKey(1)
