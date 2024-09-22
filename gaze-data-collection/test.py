# import cv2
# import numpy as np

# # 빈 이미지 생성 (검은 배경)
# img = np.zeros((512, 512, 3), np.uint8)

# # 이미지를 화면에 표시
# cv2.imshow("Test Window", img)

# # 키 입력을 대기하고 키 값을 출력
# while True:
#     key = cv2.waitKey(0)  # 키 입력을 무한정 대기
#     print(f"Pressed key: {key}")  # 키 값을 출력

#     if key == ord('q'):  # 'q'를 누르면 종료
#         break

# # 창 닫기
# cv2.destroyAllWindows()
import cv2

# 웹캠 열기
cap = cv2.VideoCapture(1)  # 0은 기본 웹캠 장치를 의미
for i in range(10):  # 0부터 9까지 장치 인덱스 테스트
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camera {i} is available")
    else:
        print(f"Camera {i} is not available")


if not cap.isOpened():
    print("Cannot open camera")
    exit()

# 프레임의 너비와 높이 가져오기
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"Frame width: {width}, Frame height: {height}")

# 비디오 읽기 및 종료 (q를 눌러 종료)
while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
