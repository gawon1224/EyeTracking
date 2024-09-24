from l2cs import select_device, Pipeline
from gaze_pipeline_module import parse_args, run_demo
import pathlib
import queue
import torch.backends.cudnn as cudnn

import os
import configparser


# AI.ini 파일을 경로에서 불러와 읽습니다.
OUTPUT_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'AI.ini')
output = configparser.ConfigParser()
ret = output.read(OUTPUT_FILE_PATH)


# 만약 실패하면 프로그램을 종료합니다.
if not ret:
    print(f"{OUTPUT_FILE_PATH} 파일이 존재하지 않거나, 읽을 수 없습니다.")
    exit(0)


# AI모델 로딩 작업을 수행합니다.
#
#
print('모델 로딩 완료')


# AI모델 로딩이 완료되면, AI.ini의 loaded 항목을 1로 변경합니다.
output.set('MODEL', 'loaded', '1')
with open(OUTPUT_FILE_PATH, 'w') as ini:
    output.write(ini)


# 메인 루프
# 연속적인 작업을 위해 루프를 작성합니다.
while True:

    # 루프 내에서 실시간으로 AI.ini를 확인합니다.
    ret = output.read(OUTPUT_FILE_PATH)
    if not ret:
        break

    # 명령어를 표준 입력으로 받아들입니다.
    user_input = input()

    # # 입력이 'face'라면, 얼굴 인식을 수행합니다.
    # if user_input == 'face' and output['FACE']['recognized'] == '0':
    #     # 얼굴 인식 수행
    #     #
    #     #
    #     if faceDetected:
    #         # 얼굴이 인식되었다면, AI.ini에 정보를 기록합니다.
    #         print("얼굴이 인식되었습니다.")
    #         output.set('FACE', 'recognized', '1')
    #         output.set('FACE', 'age', '20대')
    #         output.set('FACE', 'sex', 'female')
            
    #         with open(OUTPUT_FILE_PATH, 'w') as ini:
    #             output.write(ini)
    #     else:
    #         print('얼굴이 인식되지 않았습니다')

    # # 입력이 'voice'라면, 음성 인식을 수행합니다.
    # if user_input == 'voice' and output['VOICE']['recognized'] == '0':
    #     # 음성 인식 수행
    #     #
    #     #
    #     if voiceRecognized:
    #         # 음성이 인식되었다면, AI.ini에 정보를 기록합니다.
    #         print("음성이 인식되었습니다.")
            
    #         output.set('VOICE', 'recognized', '1')
    #         output.set('VOICE', 'value', '주민등록')
            
    #         with open(OUTPUT_FILE_PATH, 'w') as ini:
    #             output.write(ini)

    #     else:
    #         print("음성이 인식되지 않았습니다.")
            
    # # 입력이 'gesture'라면, 제스쳐 인식을 수행합니다.            
    # if user_input == 'gesture' and output['GESTURE']['recognized'] == '0':
    #     # 제스쳐 인식 수행
    #     #
    #     #
    #     if gestureRecognized:
    #         # 제스쳐가 인식되었다면, AI.ini에 정보를 기록합니다.
    #         print("제스쳐가 인식되었습니다.")
    #         output.set('GESTURE', 'recognized', '1')
    #         output.set('GESTURE', 'value', '3')
            
    #         with open(OUTPUT_FILE_PATH, 'w') as ini:
    #             output.write(ini)

    #     else:
    #         print('제스쳐가 인식되지 않았습니다.')

    # 입력이 'eye'라면, 시선 추적을 수행합니다.  
    if user_input == 'eye' and output['EYE']['recognized'] == '0':
        # 시선 추적 수행
        # main.py에서 gaze_pipeline을 정의한 후 함수 호출

        # args = parse_args()

        # cudnn.enabled = True
        # arch=args.arch
        # cam = args.cam_id
        # snapshot_path = args.snapshot

        # # frame_num과 큐 정의
        # frame_num = 0
        # q = queue.Queue()
        # CWD = pathlib.Path.cwd()
        # gaze_pipeline = Pipeline(
        #     weights=CWD / 'models' / 'L2CSNet_gaze360.pkl',
        #     arch='ResNet50',
        #     device = select_device(args.device, batch_size=1)
        # )
        

        # # 실시간으로 프레임을 처리하면서 new_x, new_y 좌표를 반환받음
        # # for new_x, new_y in process_gaze_pipeline(gaze_pipeline, 0, frame_num, q):
        # #     print(f"Processed coordinates: x={new_x}, y={new_y}")
        # new_x, new_y = process_gaze_pipeline(gaze_pipeline, 0, frame_num, q)
        # print(new_x, new_y)

        last_new_x, last_new_y = None, None
        # 초기 eyeTracked 상태
        eyeTracked = False

        for new_x, new_y in run_demo():  # demo.py의 run_demo 함수 호출
        # new_x, new_y 값을 한 프레임씩 가져옴
            print(f"Retrieved coordinates: new_x={new_x}, new_y={new_y}")
            last_new_x, last_new_y = new_x, new_y  # 마지막 값을 저장    

            # new_x와 new_y가 None이 아닌 경우 eyeTracked를 True로 설정하는 조건문
            if last_new_x is not None and last_new_y is not None:
                eyeTracked = True
            else:
                eyeTracked = False

            # eyeTracked 상태를 출력 (선택적)
            print(f"Eye Tracked: {eyeTracked}")
        
            if eyeTracked:
                # 시선이 인식되었다면, AI.ini에 정보를 기록합니다.
                print("시선이 인식되었습니다.")
                output.set('EYE', 'recognized', '1')
                output.set('EYE', 'x', str(last_new_x))
                output.set('EYE', 'y', str(last_new_y))
                
                with open(OUTPUT_FILE_PATH, 'w') as ini:
                    output.write(ini)
                
                eyeTracked = False
                    
            else :
                print('시선이 인식되지 않았습니다.')

        # 입력이 'quit'라면, 루프를 빠져 나갑니다.
        if user_input == 'quit':
            break


print("프로그램이 종료되었습니다.")

