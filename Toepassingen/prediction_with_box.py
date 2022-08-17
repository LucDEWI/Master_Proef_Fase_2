from src import Functions as func
import cv2
import numpy as np
import scipy.ndimage as ndimage
from utils.dataset_processing.grasp import GraspRectangles, detect_grasps
import torch
import os
import pyrealsense2 as rs
from utils.dataset_processing import evaluation


# hier kan een predictie gemaakt worden op een scène voor de camera met een doos

# inalden van model 

model = torch.load('C:\models_test\_83_ggcnn2_met_ruis_epoch_20_val-batches_125\epoch_08_iou_0.74')
device = torch.device("cuda")
datapath=os.path.abspath(os.path.join(os.getcwd(), os.pardir))+'\data'

# inladen van data doos
box = np.load(datapath+'\e_box.npy')

#intrinsieke en extrinsieke gegevens importeren
mtx=np.load(datapath+'\intrinsics.npy')
dist=np.load(datapath+'\distortion.npy')
ext=np.load(datapath+'\extrinsics.npy')
rvecs=np.load(datapath+'\extrinsic_rvecs.npy')
tvecs=np.load(datapath+'\extrinsic_tvecs.npy')

# verwerken van het dieptebeeld
def process_depth_image(depth, crop_size, out_size=300, return_mask=True, crop_y_offset=0):
    imh, imw = depth.shape


    # croppen van het beeld
    depth_crop = depth[(imh - crop_size) // 2 - crop_y_offset:(imh - crop_size) // 2 + crop_size - crop_y_offset,
                           (imw - crop_size) // 2:(imw - crop_size) // 2 + crop_size]
    # depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)

    # Inpainten van de grens
    

    depth_crop = cv2.copyMakeBorder(depth_crop, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    depth_nan_mask = np.isnan(depth_crop).astype(np.uint8)


    depth_crop[depth_nan_mask==1] = 0


    # schalen van het beeld om inpainten kunnen toe te passen
    depth_scale = np.abs(depth_crop).max()
    depth_crop = depth_crop.astype(np.float32) / depth_scale  # Has to be float32, 64 not supported.


    depth_crop = cv2.inpaint(depth_crop, depth_nan_mask, 1, cv2.INPAINT_NS)

    # terugbrengen van het beeld naar de originele waarden
    depth_crop = depth_crop[1:-1, 1:-1]
    depth_crop = depth_crop * depth_scale


    # Resize van het beeld
    depth_crop = cv2.resize(depth_crop, (out_size, out_size), cv2.INTER_AREA)

    if return_mask:

        depth_nan_mask = depth_nan_mask[1:-1, 1:-1]
        depth_nan_mask = cv2.resize(depth_nan_mask, (out_size, out_size), cv2.INTER_NEAREST)
    return depth_crop, depth_nan_mask
    # else:
    # return depth_crop

#  een voorspelling op het beeld uitvoeren
def predict(depth, process_depth=True, crop_size=300, out_size=300, depth_nan_mask=None, crop_y_offset=0, filters=(2.0, 1.0, 1.0)):
    if process_depth:
        depth, depth_nan_mask = process_depth_image(depth, crop_size, out_size=out_size, return_mask= False, crop_y_offset=crop_y_offset)

    # berekenen van de positie output
    depth = np.clip((depth - depth.mean()), -1, 1)
    depthT = torch.from_numpy(depth.reshape(1, 1, out_size, out_size).astype(np.float32)).to(device)
    with torch.no_grad():
        pred_out = model(depthT)

    points_out = pred_out[0].cpu().numpy().squeeze()
    points_out[depth_nan_mask] = 0

    # berekenen van de hoek output
    cos_out = pred_out[1].cpu().numpy().squeeze()
    sin_out = pred_out[2].cpu().numpy().squeeze()
    ang_out = np.arctan2(sin_out, cos_out) / 2.0

    width_out = pred_out[3].cpu().numpy().squeeze() * 150.0  # Scaled 0-150:0-1

    # filter toepassen op de outputs
    if filters[0]:
        points_out = ndimage.filters.gaussian_filter(points_out, filters[0])  # 3.0
    if filters[1]:
        ang_out = ndimage.filters.gaussian_filter(ang_out, filters[1])
    if filters[2]:
        width_out = ndimage.filters.gaussian_filter(width_out, filters[2])

    points_out = np.clip(points_out, 0.0, 1.0-1e-3)

    # SM
    # temp = 0.15
    # ep = np.exp(points_out / temp)
    # points_out = ep / ep.sum()

    # points_out = (points_out - points_out.min())/(points_out.max() - points_out.min())

    return points_out, ang_out, width_out, depth.squeeze()

# starten van connectie naar camera
connection = rs.pipeline()
configuration = rs.config()
configuration.enable_stream(rs.stream.color, 1920, 1080, rs.format.bgr8, 30)
configuration.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
spat_filter = rs.spatial_filter()

temp_filter = rs.temporal_filter()





# while True:
try:
    connection.start(configuration)
    align = rs.align(rs.stream.color)
    for i in range(30):
        frames = connection.wait_for_frames()
        aligned = align.process(frames)

        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()

        #depth_frame = spat_filter.process(depth_frame)
        #depth_frame = temp_filter.process(depth_frame)
        color = np.asanyarray(color_frame.get_data())
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
        # croppen van doos uit het kleurbeeld
        color = color[box[1]+50:box[1]+box[3]-50, box[0]+50:box[0]+box[2]-75]
        # resizen naar een 300 bij 300 pixel beeld
        color = cv2.resize(color, (300, 300), interpolation= cv2.INTER_LINEAR)
        depth = np.asanyarray(depth_frame.get_data())
        # croppen van doos uit het dieptebeeld
        depth_box = depth[box[1]+50:box[1]+box[3]-50, box[0]+50:box[0]+box[2]-75]
        # resizen van het beeld naar een 300 bij 300 pixel beeld
        depth_view = cv2.resize(depth_box, (300, 300), interpolation= cv2.INTER_LINEAR)
        # colormap dieptebeeld voor controle tijdens predictie
        depth_col = cv2.applyColorMap(cv2.convertScaleAbs(depth_view, alpha= 0.03), cv2.COLORMAP_JET)
        cv2.imshow('', depth_col)
        cv2.waitKey(10)
        i= i+1




finally:
    connection.stop()
    # verwerken van dieptebeeld
    depth_crop, mask = process_depth_image(depth_view, 300)
    # predictie uitvoeren op het dieptebeeld
    out = predict(depth_crop, False, depth_nan_mask= mask)
    # print(depth_crop)
    # print(depth_crop.shape)
    #print(color.shape)
    # cv2.imshow(' ', depth_crop)
    #print(out[0], out[1], out[2], out[3])

    # outputs
    pos_out = out[0]
    angle_out = out[1]
    width_out = out[2]
    depth_out = out[3]
    #print(out[0].shape, out[1].shape, out[2].shape, out[3].shape)

    # plotten van de outputs 
    evaluation.plot_output(color, depth_crop, out[0], out[1], grasp_width_img= out[2], no_grasps= 1)


    # data grasp tergbrengen naar coördinaat in het referentiestelsel
    gs = detect_grasps(pos_out, angle_out, width_out)
    center = gs[0].center
    center_real_0 = int(box[0] + center[0] * (box[2]-75 / 300))
    center_real_1 = int(box[1] + center[1] * (box[3]-50/ 300))  # deze is juist
    #print(gs[0].center)
    #print(gs[0].angle)
    #print(gs[0].length)
    #print(gs[0].width)
    #print(depth[center])

    center_real = (center_real_0, center_real_1)

    # intrinsiek transformeren naar cameracoördinaten
    xcam, ycam, zcam = func.intrinsictrans((center_real[1], center_real[0]), depth_crop[center[1], center[0]], mtx)
    # extrinsiek transformeren naar ruimtecoördinaten
    xworld, yworld, zworld = func.extrinsictrans(depth_crop[center[1], center[0]], xcam, ycam, zcam, ext)

    xworld = round(xworld, 2)
    yworld = round(yworld, 2)
    zworld = round(zworld, 2)
    #print(xcam, ycam, zcam)
    print(xworld, yworld, zworld)