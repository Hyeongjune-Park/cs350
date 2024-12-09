# 사람 수 감지 프로그램

이 프로젝트는 비디오 파일에서 사람 수를 실시간으로 감지하고, 감지된 사람 수를 서버로 전송하는 프로그램입니다. OpenCV를 사용하여 사람을 감지하며, Python으로 작성되었습니다.

## 요구 사항

- Python 3.x
- OpenCV
- requests

## 설치

1. **가상환경 생성 및 활성화**:
   ```bash
   conda create --name myenv python=3.8
   conda activate myenv
   ```

2. **필요한 패키지 설치**:
   ```bash
   pip install -r requirements.txt
   ```

## 사용 방법

1. 비디오 파일을 준비하고, `cv.py` 파일 내의 `video_file` 변수를 비디오 파일의 경로로 수정합니다.
   ```python
   video_file = "path/to/your/video.mp4"
   ```

2. 프로그램 실행:
   ```bash
   python cv.py
   ```

3. 비디오 창이 열리면, 감지된 사람 수와 함께 사람의 위치에 사각형이 그려진 영상을 확인할 수 있습니다. 'q' 키를 누르면 비디오 재생이 종료됩니다.

## 데이터 전송

프로그램은 감지된 사람 수를 지정된 서버로 전송합니다. 서버 URL은 `send_to_server` 함수 내에서 설정할 수 있습니다.