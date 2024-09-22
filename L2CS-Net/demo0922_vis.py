import matplotlib.pyplot as plt
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
from PIL import Image, ImageOps

from face_detection import RetinaFace
from l2cs import select_device, draw_gaze, getArch, Pipeline, render
import queue

q = queue.Queue()
CWD = pathlib.Path.cwd()

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Gaze evalution using model pretrained with L2CS-Net on Gaze360.')
    parser.add_argument(
        '--device',dest='device', help='Device to run model: cpu or gpu:0',
        default="cpu", type=str)
    parser.add_argument(
        '--snapshot',dest='snapshot', help='Path of model snapshot.', 
        default='output/snapshots/L2CS-gaze360-_loader-180-4/_epoch_55.pkl', type=str)
    parser.add_argument(
        '--cam',dest='cam_id', help='Camera device id to use [0]',  
        default=0, type=int)
    parser.add_argument(
        '--arch',dest='arch',help='Network architecture, can be: ResNet18, ResNet34, ResNet50, ResNet101, ResNet152',
        default='ResNet50', type=str)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    arch=args.arch
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

    # Initialize real-time plot
    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1600)  # Assuming the screen is 1600 pixels wide
    ax.set_ylim(0, 1200)  # Assuming the screen is 1200 pixels tall
    scatter, = ax.plot([], [], 'ro')  # Create scatter plot for gaze points

    with torch.no_grad():
        coord_list = []
        init_x = []
        init_y=[]
        sigma = 25
        blob_intensity = 1.0
        x_indices = np.arange(1600)
        y_indices = np.arange(1200)
        X, Y = np.meshgrid(x_indices, y_indices)
        c = []
        M = np.ndarray((1200, 1600))
        demo_img = np.ones((1600, 1200))
        coordinates = []

        while True:
            success, frame = cap.read()    
            frame = cv2.flip(frame, 1)
            start_fps = time.time()  

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)

            # Process frame
            results, x_min, x_max, y_min, y_max, bbox_width, bbox_height = gaze_pipeline.step(frame)
            frame = render(frame, results)

            # Rotation matrix part (remains unchanged)
            p = results.pitch
            y = results.yaw
            r = 0
            origin = np.array([0, 0, 30])

            cy = np.cos(y)
            sy = np.sin(y)
            cr = np.cos(r)
            sr = np.sin(r)
            cp = np.cos(p)
            sp = np.sin(p)

            R_x = np.array([[1, 0, 0],
                            [0, cp[0], -1*sp[0]],
                            [0, sp[0], cp[0]]])

            R_y = np.array([[cy[0], 0, sy[0]],
                            [0, 1, 0],
                            [-1*sy[0], 0, cy[0]]])

            R_z = np.array([[cr, -1*sr, 0],
                            [sr, cr, 0],
                            [0, 0, 1]])

            rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
            moved_point = np.dot(rotation_matrix, origin)
            factor = (40/2.54*138) / moved_point[2]
            new_point = np.array([moved_point[0] * factor, moved_point[1] * factor, moved_point[2] * factor])

            new_x = int(new_point[1]) + int((x_min + bbox_width / 2.0) / 640 * 1600)
            new_y = int(-1 * new_point[0]) + int((y_min + bbox_height / 3.0) / 480 * 1200)
            new_z = int(new_point[2])

            if frame_num <= 30:
                cv2.putText(demo_img, 'See WebCam', (x_min, y_max), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2, cv2.LINE_AA)
                cv2.rectangle(demo_img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
                if frame_num >= 10:
                    init_x.append(int(new_point[1]) - int(1600 / 2) + int((x_min + bbox_width / 2.0) / 640 * 1600))
                    init_y.append(int(-1 * new_point[0]) + int((y_min + bbox_height / 3.0) / 480 * 1200))
                    cv2.circle(demo_img, (int(init_x[-1]), int(init_y[-1])), 15, (0, 0, 225), -1)
            elif frame_num > 30:
                new_x = new_x - int(np.mean(init_x))
                new_y = new_y - int(np.mean(init_y))
                coord_list.append([new_x, new_y])
                point_list.append([new_x, new_y])

                # Real-time plotting of points
                scatter.set_xdata([p[0] for p in point_list])  # Update X data
                scatter.set_ydata([p[1] for p in point_list])  # Update Y data
                fig.canvas.draw()
                fig.canvas.flush_events()  # Refresh the plot

                blob_center = (new_x, new_y)
                gaussian_blob = blob_intensity * np.exp(-((X - blob_center[0]) ** 2 + (Y - blob_center[1]) ** 2) / (2 * sigma ** 2))
                q.put(gaussian_blob)
                M += gaussian_blob
                if q.qsize() >= 20:
                    M -= q.get()

                coordinates = []
                sorted_indices = np.argsort(M.flatten())[::-1]
                top_indices = sorted_indices[:10]
                for index in top_indices:
                    x = index % M.shape[1]
                    y = index // M.shape[1]
                    coordinates.append([int(x), int(y)])
                c.append(coordinates)
                cv2.circle(demo_img, (coordinates[0][0], coordinates[0][1]), 15, (0, 0, 255), -1)
                cv2.circle(demo_img, (new_x, new_y), 15, (255, 0, 0), -1)

            myFPS = 1.0 / (time.time() - start_fps)
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)

            cv2.imshow("Demo", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_num += 1
            success, frame = cap.read()

        df = pd.DataFrame(point_list)
        df.to_csv('point_data.csv', index=False)

