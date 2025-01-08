import numpy as np
import cv2
import mediapipe as mp
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Mediapipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 랜드마크에서 각도 계산 함수
def extract_landmarks(image):
    # Mediapipe를 사용하여 랜드마크 추출
    results = pose.process(image)
    landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            landmarks.append([landmark.x, landmark.y, landmark.z])  # 각 랜드마크의 x, y, z 좌표
    return np.array(landmarks).flatten()

# 학습 데이터를 위한 X, Y 생성
X = []  # 입력 데이터 (랜드마크 좌표)
Y = []  # 라벨 (동작 종류)

# 볼링 영상 파일 경로 리스트
video_paths = [
    "C:/Users/wndhk/vision/bowling1.mp4", 
    "C:/Users/wndhk/vision/bowling2.mp4",
    "C:/Users/wndhk/vision/bowling3.mp4",
    "C:/Users/wndhk/vision/bowling4.mp4",
    "C:/Users/wndhk/vision/bowling5.mp4",
    "C:/Users/wndhk/vision/bowling6.mp4",
    "C:/Users/wndhk/vision/bowling7.mp4",
    "C:/Users/wndhk/vision/bowling8.mp4",
    "C:/Users/wndhk/vision/bowling9.mp4"
]

# 각 영상에 대한 라벨 (예: 0: 스윙 준비, 1: 공 던지기, 2: 피니쉬 동작)
labels = [0, 1, 2, 0, 1, 2, 0, 1, 2]

# 영상에서 랜드마크 추출 및 라벨링
for i, video_path in enumerate(video_paths):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # 랜드마크 추출
        landmarks = extract_landmarks(frame)
        if landmarks.size > 0:  # 랜드마크가 있는 경우만 추가
            X.append(landmarks)
            Y.append(labels[i])  # 해당 영상의 라벨 추가

    cap.release()

# 데이터 배열로 변환
X = np.array(X)
Y = np.array(Y)

# 학습 데이터와 테스트 데이터로 분할
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 모델 훈련 (RandomForest 예시)
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, Y_train)

# 모델 평가
accuracy = model.score(X_test, Y_test)
print(f"모델 정확도: {accuracy:.2f}")

# 모델 저장
joblib.dump(model, 'C:/Users/wndhk/vision/bowling_pose_model.pkl')
