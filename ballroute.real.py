import cv2
import numpy as np

#됐다 ㅅ발
# 공의 궤적 구하기, 3D 궤적을 추적하기 위한 기본 코드
# 영상 불러오기
cap = cv2.VideoCapture("C:/Users/wndhk/vision/ball2.mp4")

# 공의 색상 범위 설정 (HSV 색상 공간)
lower_color = np.array([30, 150, 50])  # 색상 범위의 하한값 (예: 초록색 공)
upper_color = np.array([90, 255, 255])  # 색상 범위의 상한값

# 카메라 내부 파라미터 (가상의 예시)
focal_length = 1000  # 초점 거리 (임의 설정)
center = (640, 360)  # 화면의 중심점 (가상의 예시)

# 공의 경로를 저장할 리스트
path_points_2D = []  # 2D 좌표로 궤적을 저장

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

            # 공의 3D 위치 추정 (임시로 깊이 값은 반경을 이용하여 추정)
            z = radius * 10  # 반경을 이용하여 대략적인 깊이 추정 (가상의 예시)
            # 2D -> 3D 변환 (단순 원근법)
            X = (x - center[0]) * z / focal_length
            Y = (y - center[1]) * z / focal_length
            Z = z  # 깊이 값은 z로 설정

            # 2D 좌표로 변환 후 경로 저장 (단순히 화면에 표시할 때는 깊이 값은 무시)
            path_points_2D.append((int(x), int(y)))

    # 경로 그리기 (2D로 표시)
    for i in range(1, len(path_points_2D)):
        cv2.line(frame, path_points_2D[i-1], path_points_2D[i], (0, 0, 255), 2)

    # 영상 출력
    cv2.imshow("Ball Tracking", frame)
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 영상 리소스 해제
cap.release()
cv2.destroyAllWindows()
