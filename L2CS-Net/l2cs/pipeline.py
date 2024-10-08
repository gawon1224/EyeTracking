import pathlib
from typing import Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from dataclasses import dataclass
from face_detection import RetinaFace

from .utils import prep_input_numpy, getArch
from .results import GazeResultContainer


class Pipeline:

    def __init__(
        self, 
        weights: pathlib.Path, 
        arch: str,
        device: str = 'cpu', 
        include_detector:bool = True,
        confidence_threshold:float = 0.5
        ):

        # Save input parameters
        self.weights = weights
        self.include_detector = include_detector
        self.device = device
        self.confidence_threshold = confidence_threshold

        # Create L2CS model
        self.model = getArch(arch, 90)
        self.model.load_state_dict(torch.load(self.weights, map_location=device))
        self.model.to(self.device)
        self.model.eval()

        # Create RetinaFace if requested
        if self.include_detector:

            if device.type == 'cpu':
                self.detector = RetinaFace()
            else:
                self.detector = RetinaFace(gpu_id=device.index)

            self.softmax = nn.Softmax(dim=1)
            self.idx_tensor = [idx for idx in range(90)]
            self.idx_tensor = torch.FloatTensor(self.idx_tensor).to(self.device)

    def step(self, frame: np.ndarray) -> GazeResultContainer:

        # Creating containers
        face_imgs = []
        bboxes = []
        landmarks = []
        scores = []

        if self.include_detector:
            faces = self.detector(frame)

            if faces is not None: 
                for box, landmark, score in faces:

                    # Apply threshold
                    if score < self.confidence_threshold:
                        continue

                    # Extract safe min and max of x,y
                    x_min=int(box[0])
                    if x_min < 0:
                        x_min = 0
                    y_min=int(box[1])
                    if y_min < 0:
                        y_min = 0
                    x_max=int(box[2])
                    y_max=int(box[3])

                    # bbox size 계산 추가
                    bbox_width = x_max - x_min
                    bbox_height = y_max - y_min

                    # Crop image
                    img = frame[y_min:y_max, x_min:x_max]
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (224, 224))
                    face_imgs.append(img)

                    # Save data
                    bboxes.append(box)
                    landmarks.append(landmark)
                    scores.append(score)

                # Predict gaze
                pitch, yaw = self.predict_gaze(np.stack(face_imgs))

            else:

                pitch = np.empty((0,1))
                yaw = np.empty((0,1))

        else:
            pitch, yaw = self.predict_gaze(frame)

        # Save data
        results = GazeResultContainer(
            pitch=pitch,
            yaw=yaw,
            bboxes=np.stack(bboxes),
            landmarks=np.stack(landmarks),
            scores=np.stack(scores)
        )

        # ## 회전행렬 사용 부분 추가
        # p = pitch
        # y = yaw
        # r = 0

        # origin = np.array([0,0,30]) # 30이 떨어진 거리 말하는듯

        # cy = np.cos(y)
        # sy = np.sin(y)
        # cr = np.cos(r)
        # sr = np.sin(r)
        # cp = np.cos(p)
        # sp = np.sin(p)

        # R_x = np.array([[1, 0, 0],
        #                     [0, cp, -sp],
        #                     [0, sp, cp]])

        # R_y = np.array([[cy, 0, sy],
        #                     [0, 1, 0],
        #                     [-sy, 0, cy]])

        # R_z = np.array([[cr, -sr, 0],
        #                     [sr, cr, 0],
        #                     [0, 0, 1]])

        # ## 회전 행렬들의 곱으로 최종 회전 행렬 계산
        # # 고민 : demo06014.py에서 새로 추가된 변수들(coord_list, point_list등 프레임이 새로 들어옴에 따라 계속 업데이트 되는 값) 선언을 이 파일에서 해야할지 아니면 demo.py에서 해야할지?
        # rotation_matrix = np.dot(R_z, np.dot(R_y, R_x))
        # moved_point = np.dot(rotation_matrix, origin)
        # factor = (40/2.54*138)/ moved_point[2]
        # new_point = np.array([moved_point[0]*factor, moved_point[1]*factor,moved_point[2]*factor])

        # new_x = int(new_point[1]) + int((x_min + bbox_width/2.0)/640 * 1600)
        # new_y = int(-1*new_point[0]) + int((y_min+bbox_height/3.0)/ 480 * 1200)
        # new_z = int(new_point[2])
        # if(frame_num<=30):
        #     cv2.putText(demo_img, 'See WebCam', (x_min, y_max),cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0),2, cv2.LINE_AA)
        #     cv2.rectangle(demo_img, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
        #     if(frame_num>=10):
        #         init_x.append(int(new_point[1])-int(1600/2)+int((x_min + bbox_width/2.0)/640 * 1600))
        #         init_y.append(int(-1*new_point[0])+int((y_min+bbox_height/3.0)/ 480 * 1200))
        #         cv2.circle(demo_img, (int(init_x[-1]), int(init_y[-1])),15,(0, 0,225),-1)                         
        # elif(frame_num>30):
        #     new_x = new_x - int(np.mean(init_x))
        #     new_y = new_y -int(np.mean(init_y))
        #     coord_list.append([new_x, new_y])
        #     point_list.append([new_x,new_y])
        #     blob_center = (new_x,new_y)
        #     gaussian_blob = blob_intensity * np.exp(-((X - blob_center[0]) ** 2 + (Y - blob_center[1]) ** 2) / (2 * sigma ** 2))
        #     q.put(gaussian_blob)
        #     M += gaussian_blob
        #     if q.qsize() >=20:
        #         M-= q.get()
        #     coordinates = []
        #     sorted_indices = np.argsort(M.flatten())[::-1]
        #     top_indices = sorted_indices[:10]
        #     for index in top_indices:
        #         x = index % M.shape[1]
        #         y = index // M.shape[1]
        #         coordinates.append([int(x),int(y)])
        #     c.append(coordinates)
        #     cv2.circle(demo_img, (coordinates[0][0],coordinates[0][1]),15,(0, 0, 255),-1)
        #     cv2.circle(demo_img, (new_x,new_y),15,(255, 0, 0),-1)

        return results, x_min, x_max, y_min, y_max, bbox_width, bbox_height

    def predict_gaze(self, frame: Union[np.ndarray, torch.Tensor]):
        
        # Prepare input
        if isinstance(frame, np.ndarray):
            img = prep_input_numpy(frame, self.device)
        elif isinstance(frame, torch.Tensor):
            img = frame
        else:
            raise RuntimeError("Invalid dtype for input")
    
        # Predict 
        gaze_pitch, gaze_yaw = self.model(img)
        pitch_predicted = self.softmax(gaze_pitch)
        yaw_predicted = self.softmax(gaze_yaw)
        
        # Get continuous predictions in degrees.
        pitch_predicted = torch.sum(pitch_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted.data * self.idx_tensor, dim=1) * 4 - 180
        
        pitch_predicted= pitch_predicted.cpu().detach().numpy()* np.pi/180.0
        yaw_predicted= yaw_predicted.cpu().detach().numpy()* np.pi/180.0

        return pitch_predicted, yaw_predicted