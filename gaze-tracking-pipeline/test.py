### monitor 해상도 코드
# from screeninfo import get_monitors

# try:
#     monitor = get_monitors()[0]  # 첫 번째 모니터의 해상도를 가져옴
#     monitor_width = monitor.width
#     monitor_height = monitor.height
#     print(f"Monitor dimensions: {monitor_width}x{monitor_height}")
# except Exception as e:
#     print(f"Could not retrieve monitor dimensions: {e}")
#     # # 수동으로 모니터 해상도 설정
#     # monitor_width = 1920
#     # monitor_height = 1080
#     # print(f"Using manual monitor dimensions: {monitor_width}x{monitor_height}")

import collections
import time
from argparse import ArgumentParser

import albumentations as A
import cv2
import mediapipe as mp
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

from model import Model
from mpii_face_gaze_preprocessing import normalize_single_image
from utils import get_camera_matrix, get_face_landmarks_in_ccs, gaze_2d_to_3d, ray_plane_intersection, plane_equation, get_monitor_dimensions, get_point_on_screen
from visualization import Plot3DScene
from webcam import WebcamSource


objp = np.zeros((9*6,3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)
objpoints = []
imgpoints = []

img = cv2.imread(r"C:\Users\uos\Desktop\EyeTracking\gaze-tracking-pipeline\callibration_img\capture_2024-09-12_16-13-20_2.png")
patternSize = (9, 6)
_img = cv2.resize(img, dsize = (640,480), interpolation = cv2.INTER_AREA)
gray = cv2.cvtColor(_img, cv2.COLOR_BGR2GRAY)

ret, corners = cv2.findChessboardCorners(gray, patternSize, None)

if ret == True:
    objpoints.append(objp)

    corners2 = cv2.cornerSubPix(gray, corners, (10,10), (-1,-1), criteria)
    imgpoints.append(corners2)

    img = cv2.drawChessboardCorners(_img, (9,6), corners2, ret)