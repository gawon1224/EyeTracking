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
    # snapshot_path = args.snapshot

    gaze_pipeline = Pipeline(
        weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device = select_device(args.device, batch_size=1)
    )
     
    cap = cv2.VideoCapture(cam)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        raise IOError("Cannot open webcam")
    
    # 추가
    frame_num = 0
    point_list =[]
    ####

    with torch.no_grad():
        ## 전역변수 추가
        coord_list = []
        init_x = []
        init_y=[]
        sigma = 25  # 가우시안 Blob의 표준 편차
        blob_intensity = 1.0  # Blob의 강도
        x_indices = np.arange(2560)
        y_indices = np.arange(1600)
        X, Y = np.meshgrid(x_indices, y_indices)
        c =[]
        M = np.ndarray((1600,2560))
        # demo_img = np.ones((2560,1600))
        #demo_img = np.zeros((2560,1600))
        coordinates = []

        while True:
            # Get frame
            success, frame = cap.read()    
            frame=cv2.flip(frame,1) # 좌우반전 추가 ; 필요한진 몰겠음
            # 프레임마다 demo_img를 초기화
            demo_img = np.ones((2560, 1600))  # 매 프레임마다 새로 초기화

            start_fps = time.time()  

            if not success:
                print("Failed to obtain frame")
                time.sleep(0.1)

            # Process frame
            results, x_min, x_max, y_min, y_max, bbox_width, bbox_height = gaze_pipeline.step(frame)
            
            # Visualize output
            frame = render(frame, results)
            # rotation matrix 적용 -> frame 여러개 값 누적해야하므로 pipeline.py 말고 여기에 추가하는 게 맞을듯?
            ## 회전행렬 사용 부분 추가
            p = results.pitch
            y = results.yaw
            r = 0

            origin = np.array([0,0,50]) # 원랜 30, 카메라로부터 사람이 떨어진 거리 말하는듯?, 50으로 늘리니까 더 잘됨

            cy = np.cos(y)
            sy = np.sin(y)
            cr = np.cos(r)
            sr = np.sin(r)
            cp = np.cos(p)
            sp = np.sin(p)
            # print('cy',cy[0],'sy',sy[0],'cr',cr,'sr',sr,'cp',cp[0],'sp',sp[0])

            R_x = np.array([[1, 0, 0],
                            [0, cp[0], -1*sp[0]],
                            [0, sp[0], cp[0]]])

            R_y = np.array([[cy[0], 0, sy[0]],
                            [0, 1, 0],
                            [-1*sy[0], 0, cy[0]]])

            R_z = np.array([[cr, -1*sr, 0],
                            [sr, cr, 0],
                            [0, 0, 1]])
            
            # print(R_x,R_y,R_z)
            ## 회전 행렬들의 곱으로 최종 회전 행렬 계산
            # 고민 : demo06014.py에서 새로 추가된 변수들(coord_list, point_list등 프레임이 새로 들어옴에 따라 계속 업데이트 되는 값) 선언을 이 파일에서 해야할지 아니면 demo.py에서 해야할지?
            rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
            moved_point = np.dot(rotation_matrix, origin)
            factor = (40/2.54*138)/ moved_point[2]
            new_point = np.array([moved_point[0]*factor, moved_point[1]*factor,moved_point[2]*factor])

            new_x = int(new_point[1]) + int((x_min + bbox_width/2.0)/640 * 2560)
            new_y = int(-1*new_point[0]) + int((y_min+bbox_height/3.0)/ 480 * 1600)
            new_z = int(new_point[2])

            if(frame_num<=20):  # 원랜 30 수정 ㄱㄱ - 첨에 see webcam 문구를 몇 프레임동안 띄울지인듯
                cv2.putText(demo_img, 'See WebCam', (x_min, y_max),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),2, cv2.LINE_AA)
                cv2.rectangle(demo_img, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                if(frame_num>=10):
                    init_x.append(int(new_point[1])-int(2560/2)+int((x_min + bbox_width/2.0)/640 * 2560))
                    init_y.append(int(-1*new_point[0])+int((y_min+bbox_height/3.0)/ 480 * 1600))
                    cv2.circle(demo_img, (int(init_x[-1]), int(init_y[-1])),15,(0, 0,225),-1)                         
            elif(frame_num>20):
                new_x = new_x - int(np.mean(init_x))
                new_y = new_y -int(np.mean(init_y))
                coord_list.append([new_x, new_y])
                point_list.append([new_x,new_y])
                
                blob_center = (new_x,new_y)
                gaussian_blob = blob_intensity * np.exp(-((X - blob_center[0]) ** 2 + (Y - blob_center[1]) ** 2) / (2 * sigma ** 2))
                q.put(gaussian_blob)
                M += gaussian_blob
                if q.qsize() >= 5:  #20 -> 10 -> 5 memory 이슈, 숫자 줄일수록 잔류 안 함, 2는 너무 작은거 같기도?
                    M-= q.get()
                coordinates = []
                sorted_indices = np.argsort(M.flatten())[::-1]
                top_indices = sorted_indices[:10]
                for index in top_indices:
                    x = index % M.shape[1]
                    y = index // M.shape[1]
                    coordinates.append([int(x),int(y)])
                c.append(coordinates)
                cv2.circle(demo_img, (coordinates[0][0],coordinates[0][1]),15,(0, 0, 255),-1)
                cv2.circle(demo_img, (new_x,new_y),15,(255, 0, 0),-1)

            
            myFPS = 1.0 / (time.time() - start_fps)

            # 첫 번째 창: frame
            cv2.putText(frame, 'FPS: {:.1f}'.format(myFPS), (10, 20), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow("Original Frame", frame)

            # 두 번째 창: demo_img
            # # Normalize the accumulated Gaussian Blob matrix M to a range of [0, 255]
            M_normalized = cv2.normalize(M, None, 0, 255, cv2.NORM_MINMAX)
            # Convert to uint8 for applying color map (required for cv2.applyColorMap)
            M_uint8 = M_normalized.astype(np.uint8)
            # Apply a color map to the normalized matrix to create a heatmap effect
            heatmap = cv2.applyColorMap(M_uint8, cv2.COLORMAP_JET)
            # Convert demo_img to uint8 type (from CV_64F)
            demo_img_uint8 = demo_img.astype(np.uint8)
            # Convert the grayscale demo_img to a 3-channel BGR image
            demo_img_colored = cv2.cvtColor(demo_img_uint8, cv2.COLOR_GRAY2BGR)
            # Resize the heatmap to match the size of demo_img_colored (if necessary)
            heatmap_resized = cv2.resize(heatmap, (demo_img_colored.shape[1], demo_img_colored.shape[0]))
            # Add the heatmap to the demo image
            # demo_img_with_heatmap = cv2.addWeighted(demo_img_colored, 0.6, heatmap_resized, 0.4, 0)
            demo_img_with_heatmap = cv2.addWeighted(demo_img_colored, 0.3, heatmap_resized, 0.7, 0)

            # Create a window before resizing it
            cv2.namedWindow('Demo Image with Heatmap', cv2.WINDOW_NORMAL)
            # Resize the window to the desired screen dimensions
            screen_width, screen_height = [2560, 1600]
            cv2.resizeWindow('Demo Image with Heatmap', screen_width, screen_height)
            # Show the resulting image with heatmap
            cv2.imshow("Demo Image with Heatmap", demo_img_with_heatmap)



            # 공통 키 입력 처리
            if cv2.waitKey(1) & 0xFF == 27:  # ESC를 눌렀을 때 종료
                # print(c) 
                cv2.destroyAllWindows()
                break

            frame_num += 1
            success, frame = cap.read()


        df = pd.DataFrame(point_list)
        df.to_csv('point_data.csv',index = False)  

    
