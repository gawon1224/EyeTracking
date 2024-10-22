# Eye-tracking method for knowing where people are looking on the monitor
## Wrap up Report
[eye tracking wrap up report.pdf]([eye tracking report.pdf](https://github.com/user-attachments/files/17470304/eye.tracking.report.pdf)
)
)
## Pipeline
![Pipeline](![eye tracking pipeline](https://github.com/user-attachments/assets/acafd894-4b6e-4c53-b4d6-ed983a758fd7))


## Demo Result
https://github.com/user-attachments/assets/ca308098-6919-4348-9618-f666676346b4

## Usage
### Install packages and prepare 
1. git clone https://github.com/gawon1224/EyeTracking.git
2. conda 가상환경 / local에 `pip install git+https://github.com/edavalosanaya/L2CS-Net.git@main`
3. pretrained model 드라이브 링크를 통해 `L2CSNet_gaze360.pkl` 다운로드해서 models 폴더 하위에 업로드
4. `demo0922.py`실행
    - `models/L2CSNet_gaze360.pkl` 모델 pkl 저장된 경로는 데스크탑 경로에 따라 수정 필요
```python
# 실행 명령
python demo0922.py --snapshot models/L2CSNet_gaze360.pkl
```
