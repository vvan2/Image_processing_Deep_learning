import cv2
import mediapipe as mp
import math

# Mediapipe Pose 초기화
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 두 랜드마크 간 각도 계산 함수
def calculate_angle(a, b, c):
    """a, b, c는 랜드마크의 (x, y) 좌표 튜플"""
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )
    return abs(angle) if abs(angle) <= 180 else 360 - abs(angle)

# 동작 인식 함수
def classify_pose(landmarks, image_shape):
    """랜드마크를 분석하여 동작을 분류"""
    height, width, _ = image_shape

    # 랜드마크 좌표 가져오기
    left_hip = (landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * width,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * height)
    left_knee = (landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * width,
                 landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * height)
    left_ankle = (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * width,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * height)

    # 각도 계산 (왼쪽 다리 기준 예시)
    knee_angle = calculate_angle(left_hip, left_knee, left_ankle)

    # 동작 분류 (예: 서기, 앉기)
    if knee_angle > 160:  # 다리가 거의 펴져 있는 경우
        return "Standing"
    elif knee_angle < 100:  # 다리가 굽혀져 있는 경우
        return "Sitting"
    else:
        return "Unknown"

# 웹캠 입력
cap = cv2.VideoCapture(0)

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
            print("카메라에서 영상을 가져올 수 없습니다.")
            break

        # BGR 이미지를 RGB로 변환
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Mediapipe 처리
        results = pose.process(image)

        # 이미지 다시 쓰기 가능 설정
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크가 감지되었을 때
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

            # 동작 분석
            landmarks = results.pose_landmarks.landmark
            pose_class = classify_pose(landmarks, image.shape)
            cv2.putText(image, f"Pose: {pose_class}", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 결과 영상 출력
        cv2.imshow("Real-Time Pose Analysis", image)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
