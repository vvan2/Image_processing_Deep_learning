import cv2
import mediapipe as mp

# Mediapipe Pose 솔루션 초기화
mp_drawing = mp.solutions.drawing_utils  # 포즈 랜드마크를 그리기 위한 유틸리티
mp_pose = mp.solutions.pose  # 포즈 추적 모듈

# 웹캠 입력 받기
cap = cv2.VideoCapture(0)

# Pose 솔루션 초기화
with mp_pose.Pose(
    static_image_mode=False,       # 동영상 입력 시 False
    model_complexity=1,            # 모델 복잡도 (0, 1, 2)
    smooth_landmarks=True,         # 랜드마크 부드럽게
    min_detection_confidence=0.5,  # 최소 검출 신뢰도
    min_tracking_confidence=0.5    # 최소 추적 신뢰도
) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("카메라에서 영상을 가져올 수 없습니다.")
            break

        # BGR 이미지를 RGB로 변환 (Mediapipe는 RGB 입력 필요)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False  # 성능 향상을 위해 쓰기 비활성화

        # Mediapipe로 포즈 처리
        results = pose.process(image)

        # 이미지 쓰기 가능으로 변경
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 랜드마크가 감지되었을 경우
        if results.pose_landmarks:
            # 랜드마크와 연결선 그리기
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,           # 랜드마크 데이터
                mp_pose.POSE_CONNECTIONS,         # 랜드마크 연결선
                mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2),  # 랜드마크 스타일
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)                   # 연결선 스타일
            )

            # 각 랜드마크의 좌표 출력 (예: 코, 어깨, 엉덩이 등)
            for id, landmark in enumerate(results.pose_landmarks.landmark):
                height, width, _ = image.shape
                x, y = int(landmark.x * width), int(landmark.y * height)
                print(f"랜드마크 {id}: ({x}, {y})")

        # 결과 출력
        cv2.imshow("Mediapipe Pose", image)

        # 'q'를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# 리소스 해제
cap.release()
cv2.destroyAllWindows()
