# Eye-tracking method for knowing where people are looking on the monitor
## Wrap up Report
[eye tracking report.pdf](https://github.com/user-attachments/files/17470304/eye.tracking.report.pdf)

## Pipeline
![eye tracking pipeline](https://github.com/user-attachments/assets/2ff796d5-0ff7-4f22-b6f0-9455c1d725ce)

## Demo Result
https://github.com/user-attachments/assets/ca308098-6919-4348-9618-f666676346b4

## Usage
### Install packages and prepare 
1. git clone https://github.com/gawon1224/EyeTracking.git
2. conda 가상환경 또는 local에 `pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main` 을 실행합니다.
3. [pretrained model download 링크](https://drive.google.com/drive/folders/17p6ORr-JQJcw-eYtG2WGNiuS_qVKwdWd)를 통해 `L2CSNet_gaze360.pkl` 다운로드하여 models 폴더 하위에 업로드합니다.
4. `demo0922.py`실행
    - 여기서 `models/L2CSNet_gaze360.pkl` 모델의 .pkl 파일이 저장된 경로는 데스크탑 경로에 맞추어 수정이 필요합니다.
```python
# 실행 명령 예시
python demo0922.py --snapshot models/L2CSNet_gaze360.pkl
```
