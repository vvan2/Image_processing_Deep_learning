import cv2
import mediapipe as mp
import math

# Mediapipe Pose 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 두 랜드마크 간 각도 계산 함수 (동일)
def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle) if abs(angle) <= 180 else 360 - abs(angle)

# 동작 인식 함수 (동일)
def classify_pose(landmarks, image_shape):
    height, width, _ = image_shape
    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height)
    left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height)
    left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height)
    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

    if knee_angle > 160:
        return "Standing"
    elif knee_angle < 100:
        return "Sitting"
    else:
        return "Unknown"

# 영상 파일 입력
# 절대 경로로 영상 확인
video_path = "C:/Users/wndhk/vision/exercise.mp4" 
cap = cv2.VideoCapture(video_path)

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
            pose_class = classify_pose(landmarks, image.shape)
            cv2.putText(image, f"Pose: {pose_class}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Video Pose Analysis", image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
