import argparse
import pathlib
import numpy as np
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

CWD = pathlib.Path.cwd()

# 모델 로드 함수 (위에서 작성한 함수 사용)
def load_model(snapshot_path, arch='ResNet50', gpu_id='0'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = getArch(arch, 90)
    print('Loading snapshot.')
    saved_state_dict = torch.load(snapshot_path, map_location=device)
    model.load_state_dict(saved_state_dict)
    model = model.to(device)
    model.eval()
    return model, device

# 이미지 전처리 함수
def preprocess_image(image):
    transformations = transforms.Compose([
        transforms.Resize(448),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    im_pil = Image.fromarray(image)
    return transformations(im_pil).unsqueeze(0)  # 배치 차원 추가

# 모델 추론 함수
def predict_gaze(model, device, img_tensor):
    softmax = torch.nn.Softmax(dim=1)
    idx_tensor = torch.FloatTensor([idx for idx in range(90)]).to(device)
    
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        gaze_pitch, gaze_yaw = model(img_tensor)
        
        pitch_predicted = softmax(gaze_pitch)
        yaw_predicted = softmax(gaze_yaw)
        
        pitch_predicted = torch.sum(pitch_predicted[0] * idx_tensor) * 4 - 180
        yaw_predicted = torch.sum(yaw_predicted[0] * idx_tensor) * 4 - 180
        
        pitch_predicted = pitch_predicted.item() * np.pi / 180.0  # 라디안 변환
        yaw_predicted = yaw_predicted.item() * np.pi / 180.0  # 라디안 변환
    
    return pitch_predicted, yaw_predicted

# 웹캠을 통해 실시간 이미지 처리 함수
def process_webcam(snapshot_path):
    # 모델 로드
    model, device = load_model(snapshot_path)
    
    # OpenCV를 이용하여 웹캠에서 영상 받기
    cap = cv2.VideoCapture(0)  # 0은 기본 카메라 장치
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("웹캠을 열 수 없습니다.")
        return

    while True:
        # 웹캠으로부터 프레임 읽기
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break
        
        # 프레임을 모델의 입력으로 전처리
        img_tensor = preprocess_image(frame)
        
        # 모델 추론
        pitch, yaw = predict_gaze(model, device, img_tensor)
        
        # 결과 출력 (여기서는 간단히 텍스트로 표시)
        cv2.putText(frame, f'Pitch: {pitch:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f'Yaw: {yaw:.2f}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # 결과를 화면에 표시
        cv2.imshow('Webcam Gaze Prediction', frame)
        
        # ESC 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == 27:
            break

    # 리소스 해제
    cap.release()
    cv2.destroyAllWindows()

# 메인 실행
if __name__ == '__main__':
    snapshot_path = 'models/L2CSNet_gaze360.pkl'
    process_webcam(snapshot_path)
