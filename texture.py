import cv2
import mediapipe as mp

# 사용자의 자세를 보고 자세를 추정하는 것 그러나, 학습 데이터가 없기때문에 일단 간단한 출력 예시
# MediaPipe Pose 모델 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 영상 불러오기
cap = cv2.VideoCapture("video_path.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 영상 BGR -> RGB로 변환
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 자세 추정
    result = pose.process(rgb_frame)
    
    if result.pose_landmarks:
        # 랜드마크 그리기
        mp.solutions.drawing_utils.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # 영상 출력
    cv2.imshow("Pose Estimation", frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 영상 리소스 해제
cap.release()
cv2.destroyAllWindows()
