import cv2
import numpy as np
import mediapipe as mp

#  open 라이브러리 함수 사용으로, 공의 궤적, 자세를 인식하는 코드 결합
# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 영상 불러오기
cap = cv2.VideoCapture("video_path.mp4")

# 공의 색상 범위 설정 (HSV 색상 공간)
lower_color = np.array([30, 150, 50])  # 색상 범위의 하한값 (예: 초록색 공)
upper_color = np.array([90, 255, 255])  # 색상 범위의 상한값

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 영상 색상 변환 (BGR -> HSV)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 색상 범위에 맞는 부분 마스크 생성
    mask = cv2.inRange(hsv, lower_color, upper_color)
    
    # 마스크를 사용하여 공 부분을 추출
    res = cv2.bitwise_and(frame, frame, mask=mask)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # 가장 큰 윤곽선 찾기 (공이 가장 큰 객체라고 가정)
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(max_contour)
        if radius > 10:
            # 공의 궤도 표시 (원 그리기)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)
    
    # MediaPipe를 사용하여 자세 추정
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = pose.process(rgb_frame)
    
    if result.pose_landmarks:
        # 랜드마크 그리기
        mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 영상 출력
    cv2.imshow("Ball and Pose Tracking", frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 영상 리소스 해제
cap.release()
cv2.destroyAllWindows()
