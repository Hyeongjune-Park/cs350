import cv2
import requests
import time
import numpy as np
import os
def count_people(frame):
    # YOLO 모델 로드
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    # 프레임을 YOLO 입력 형식으로 변환
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []
    people_count = 0

    # 감지된 객체를 처리
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            # 사람 클래스(0)만 감지
            if confidence > 0.1 and class_id == 0:  # confidence threshold
                # 감지된 사람의 위치에 사각형을 그립니다.
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Non-Maximum Suppression 적용
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.8)  # confidence threshold, NMS threshold

    # NMS 결과 처리
    if len(indices) > 0:
        for i in indices.flatten():  # flatten()을 사용하여 1D 배열로 변환
            box = boxes[i]
            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # 사각형 색상: 파란색, 두께: 2
            cv2.putText(frame, "Detected", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    return len(indices) if len(indices) > 0 else 0  # NMS 후 남은 인덱스 수를 반환

def send_to_server(count):
    # 서버로 사람 수를 전송하는 로직
    url = "http://yourserver.com/api/count"
    data = {'count': count}
    response = requests.post(url, json=data)
    return response.status_code

def process_video(video_path, interval_seconds):
    # capture 폴더 생성
    if not os.path.exists('capture'):
        os.makedirs('capture')

    # timeline.txt 파일 열기
    with open('timeline.txt', 'w') as timeline_file:
        cap = cv2.VideoCapture(video_path)
        last_time = time.time()  # 시작 시간
        people_count = 0  # people_count 변수를 초기화합니다.
        send_message = ""  # "Send to server" 메시지를 위한 변수
        send_time = 0  # 메시지를 표시한 시간을 저장할 변수

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            current_time = time.time()
            # 현재 시간 계산
            elapsed_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 초 단위
            minutes = int(elapsed_time // 60)
            seconds = int(elapsed_time % 60)

            # minutes와 seconds를 항상 2자리로 표현
            minutes_str = str(minutes).zfill(2)
            seconds_str = str(seconds).zfill(2)

            if current_time - last_time >= interval_seconds:  # 지정한 시간 간격 확인
                new_people_count = count_people(frame)  # 사람 수를 세고 변수에 저장
                if new_people_count != people_count:  # people_count가 변경된 경우
                    send_message = "Send to server"  # 메시지 설정
                    send_time = current_time  # 메시지를 표시한 시간 저장

                people_count = new_people_count  # people_count 업데이트
                send_to_server(people_count)
                last_time = current_time  # 마지막 전송 시간을 업데이트

                # timeline.txt에 기록
                timeline_file.write(f'{minutes_str}:{seconds_str}: {people_count} people observed\n')

                # 터미널에 사람 수 출력
                print(f'Time: {minutes_str}:{seconds_str}, People Count: {people_count}')

            # 감지된 사람 수를 화면에 표시
            #cv2.putText(frame, f'People Count: {people_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 현재 로그를 영상의 아랫쪽에 표시
            log_text = f'DEC. 11th. 11:{minutes:02d}:{seconds:02d}, Current People: {people_count}'
            cv2.putText(frame, log_text, (10, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # "Send to server" 메시지를 표시할 시간 체크
            if send_message and (current_time - send_time) < 1.8:
                cv2.putText(frame, send_message, (700, frame.shape[0] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # 영상 상단 중앙에 "Sample Video" 표시
            text = "Sample Video"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            cv2.putText(frame, text, (text_x, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # 결과 프레임을 화면에 표시
            cv2.imshow('Video', frame)

            if cv2.waitKey(30) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    video_file = "datasets/crowded_1.mp4"
    time_interval = 4 
    process_video(video_file, time_interval)
