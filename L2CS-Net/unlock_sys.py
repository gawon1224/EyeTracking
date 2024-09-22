import cv2
import numpy as np
import os
import time
import csv

def sharpen2D(image):
    """Sharpen an image using a custom kernel."""
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    return cv2.filter2D(image, -1, kernel)

# Initialize video captures
cap1 = cv2.VideoCapture(0)  # Webcam
cap2 = cv2.VideoCapture(1)  # USB camera

# Check if the video captures have been opened correctly
if not cap1.isOpened():
    print("첫번째 카메라를 열수 없습니다.")
if not cap2.isOpened():
    print("두번째 카메라를 열수 없습니다.")

sum_x = sum_y = 0
n = 0
state = 0
state2 = False
circles = []
starttime = None

while True:
    print(f"현재 상태: {state}")
    
    # Capture frames from both cameras
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # Flip and convert the frame from the second camera to grayscale
    frame2 = cv2.flip(frame2, 1)
    frame3 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    frame4 = cv2.cvtColor(frame3, cv2.COLOR_GRAY2BGR)

    # Resize frames
    frame3 = cv2.resize(frame3, (1280, 960))
    frame4 = cv2.resize(frame4, (1280, 960))

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(frame3, cv2.HOUGH_GRADIENT, 1, 900, param1=100, param2=30, minRadius=26, maxRadius=98)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            radius = i[2]
            
            # Draw the circles on the frame
            cv2.circle(frame4, center, radius, (0, 255, 0), 2)
            cv2.circle(frame4, center, 2, (0, 0, 255), 3)
            print(f"center: {center}")
            
            if state2 == False and state == 3 and (time.time() - starttime) >= 2:
                with open("realTime.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i[0], i[1]])
                    
            if state2 == True and state == 3 and (time.time() - starttime) >= 2:
                with open("realTime2.csv", 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([i[0], i[1]])

            if n >= 0:
                sum_x += i[0]
                sum_y += i[1]
            n += 1

    # Show the processed frame
    cv2.imshow("camera4:EYE", frame4)

    # Handling the state logic
    if state2 == False:
        if state == 0:
            # Reading start.txt for state change
            with open("start.txt", 'r') as f:
                in_line = f.readline().strip()
                if in_line == "1":
                    state = 1
                    n = 0
                    sum_x = sum_y = 0

        print(f"n은?: {n}")
        print(f"sumx: {sum_x}")
        print(f"sumy: {sum_y}")

        if n == 60:
            avg_x = sum_x // n
            avg_y = sum_y // n
            if state == 1:
                with open("startcali2.txt", 'w') as f:
                    f.write("1")
                with open("cal_1.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([avg_x, avg_y])
                state = 2
                n = -5
                sum_x = sum_y = 0
            elif state == 2:
                with open("startnextScene.txt", 'w') as f:
                    f.write("1")
                with open("cal_2.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([avg_x, avg_y])
                state = 3
                starttime = time.time()

        # Check if password.txt exists
        if state == 3 and os.path.exists("password.txt"):
            state = 0
            state2 = True
            n = sum_x = sum_y = 0

    # Handling state2 logic
    if state2 == True:
        if state == 0:
            with open("scene3-1.txt", 'r') as f:
                in_line = f.readline().strip()
                if in_line == "1":
                    state = 1
                    n = 0
                    sum_x = sum_y = 0

        print(f"n은?: {n}")
        print(f"sumx: {sum_x}")
        print(f"sumy: {sum_y}")

        if n == 60:
            avg_x = sum_x // n
            avg_y = sum_y // n
            if state == 1:
                with open("scene3-1-startcali2.txt", 'w') as f:
                    f.write("1")
                with open("scene3-cal_1.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([avg_x, avg_y])
                state = 2
                n = -5
                sum_x = sum_y = 0
            elif state == 2:
                with open("scene3-1-tonextScene.txt", 'w') as f:
                    f.write("1")
                with open("scene3-cal_2.csv", 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([avg_x, avg_y])
                state = 3
                starttime = time.time()

    if cv2.waitKey(20) == 27:  # Press ESC to exit
        break

# Release the captures and close any open windows
cap1.release()
cap2.release()
cv2.destroyAllWindows()
