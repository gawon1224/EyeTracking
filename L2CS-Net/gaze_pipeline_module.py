# demo.py

import argparse
import pathlib
import numpy as np
import pandas as pd
import cv2
import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision

from PIL import Image
from PIL import ImageOps

from face_detection import RetinaFace

from l2cs import select_device, draw_gaze, getArch, Pipeline, render

import queue

CWD = pathlib.Path.cwd()

# 전역 변수 선언
new_x = None
new_y = None

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device', dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam', dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch', dest='arch', help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

def run_demo():
    global new_x, new_y

    args = parse_args()

    cudnn.enabled = True
    arch = args.arch
    cam = args.cam_id

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=select_device(args.device, batch_size=1)
    )

    cap = cv2.VideoCapture(cam)

    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    frame_num = 0
    point_list = []
    q = queue.Queue()

    init_x = []
    init_y = []
    sigma = 25
    blob_intensity = 1.0
    x_indices = np.arange(2560)
    y_indices = np.arange(1600)
    X, Y = np.meshgrid(x_indices, y_indices)
    M = np.ndarray((1600, 2560))
    coord_list = []

    while True:
        success, frame = cap.read()
        frame = cv2.flip(frame, 1)

        demo_img = np.ones((2560, 1600))

        start_fps = time.time()

        if not success:
            print("Failed to obtain frame")
            time.sleep(0.1)

        results, x_min, x_max, y_min, y_max, bbox_width, bbox_height = gaze_pipeline.step(frame)

        frame = render(frame, results)

        p = results.pitch
        y = results.yaw
        r = 0

        origin = np.array([0, 0, 50])

        cy = np.cos(y)
        sy = np.sin(y)
        cr = np.cos(r)
        sr = np.sin(r)
        cp = np.cos(p)
        sp = np.sin(p)

        R_x = np.array([[1, 0, 0],
                        [0, cp[0], -1 * sp[0]],
                        [0, sp[0], cp[0]]])

        R_y = np.array([[cy[0], 0, sy[0]],
                        [0, 1, 0],
                        [-1 * sy[0], 0, cy[0]]])

        R_z = np.array([[cr, -1 * sr, 0],
                        [sr, cr, 0],
                        [0, 0, 1]])

        rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
        moved_point = np.dot(rotation_matrix, origin)
        factor = (40 / 2.54 * 138) / moved_point[2]
        new_point = np.array([moved_point[0] * factor, moved_point[1] * factor, moved_point[2] * factor])

        new_x = int(new_point[1]) + int((x_min + bbox_width / 2.0) / 640 * 2560)
        new_y = int(-1 * new_point[0]) + int((y_min + bbox_height / 3.0) / 480 * 1600)

        if frame_num <= 20:
            if frame_num >= 10:
                init_x.append(new_x)
                init_y.append(new_y)
        else:
            new_x = new_x - int(np.mean(init_x))
            new_y = new_y - int(np.mean(init_y))

        frame_num += 1

        # 각 프레임마다 new_x, new_y 값 업데이트됨
        yield new_x, new_y

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    for new_x, new_y in run_demo():
        print(f"new_x: {new_x}, new_y: {new_y}")
