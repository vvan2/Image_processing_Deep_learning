import cv2
import mediapipe as mp
import joblib
import numpy as np

# Mediapipe Pose 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 학습된 모델 로드
model = joblib.load('C:/Users/wndhk/vision/golf_pose_model.pkl')

# 랜드마크에서 각도 계산 함수 (동일)
def extract_landmarks(image):
    # Mediapipe를 사용하여 랜드마크 추출
    results = pose.process(image)
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])  # 각 랜드마크의 x, y, z 좌표
    return np.array(landmarks).flatten()

# 영상 파일 입력
# 절대 경로로 영상 확인
video_path = "C:/Users/wndhk/vision/golf.ln1.mp4" 
cap = cv2.VideoCapture(video_path)

# Mediapipe Pose 객체 초기화
with mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("영상 처리가 완료되었습니다.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )
            landmarks = results.pose_landmarks.landmark

            # 랜드마크 추출 및 예측
            landmark_data = np.array([extract_landmarks(frame)])
            prediction = model.predict(landmark_data)
            pose_class = prediction[0]  # 예측된 동작 라벨

            # 동작을 나타내는 텍스트 출력
            if pose_class == 0:
                status = "Driver"
            elif pose_class == 1:
                status = "Putting"
            else:
                status = "Iron"

            cv2.putText(image, f"Pose: {status}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Video Pose Analysis", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
