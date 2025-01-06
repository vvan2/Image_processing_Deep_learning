import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 데이터 전처리 함수
def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    coords = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 공 탐지 (간단히 컬러 필터 사용)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, (30, 150, 50), (50, 255, 255))  # 특정 색상 범위
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            if radius > 5:  # 최소 크기 필터링
                coords.append((x, y))

        # 프레임 크기 조정 및 채널 순서 확인
        resized_frame = cv2.resize(frame, (224, 224))  # (H, W, C)
        frames.append(resized_frame)

    cap.release()

    # 프레임 및 좌표 반환
    frames = np.array(frames)  # (T, H, W, C)
    frames = np.expand_dims(frames, axis=0)  # (1, T, H, W, C) -> Batch 추가
    coords = np.array(coords)  # (T, 2) 좌표 데이터
    print(f"Frames shape: {frames.shape}")  # 디버깅: frames shape 출력
    return frames, coords
# 모델 정의
class TrajectoryPredictor(nn.Module):
    def __init__(self):
        super(TrajectoryPredictor, self).__init__()
        # input_size를 150528로 설정
        self.lstm = nn.LSTM(input_size=150528, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# 학습 함수
def train_model(video_data, coord_data, epochs=10, lr=0.001):
    # 비디오 데이터를 Tensor로 변환
    video_data = torch.tensor(video_data, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255.0  # (B, T, C, H, W)
    coord_data = torch.tensor(coord_data, dtype=torch.float32)

    # 모델 초기화
    model = TrajectoryPredictor()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()

        # `view` 대신 `reshape` 사용
        batch_size, time_steps, channels, height, width = video_data.shape
        input_data = video_data.reshape(batch_size, time_steps, -1)  # (B, T, C*H*W)
        target_data = coord_data

        outputs = model(input_data)
        loss = criterion(outputs, target_data)
        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

    print("Training complete!")
    return model

# 학습된 모델을 파일로 저장하는 코드 추가
def save_model(model, model_path):
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")

# 모델 테스트 및 예측 코드
def predict_trajectory(model, video_data):
    model.eval()  # 평가 모드로 전환
    video_data = torch.tensor(video_data, dtype=torch.float32).permute(0, 1, 4, 2, 3) / 255.0  # (B, T, C, H, W)
    print(f"Video data shape before permute: {video_data.shape}")  # 디버깅: video_data의 shape 출력
    
    # 차원이 잘못된 경우 확인 및 디버깅
    if len(video_data.shape) != 5:
        print(f"Unexpected shape: {video_data.shape}")
        raise ValueError(f"Expected video_data to have 5 dimensions, but got {len(video_data.shape)}.")
    
    batch_size, time_steps, channels, height, width = video_data.shape
    input_data = video_data.reshape(batch_size, time_steps, -1)  # (B, T, C*H*W)
    
    with torch.no_grad():
        predicted_coords = model(input_data)
    
    return predicted_coords

# 테스트용 실행 코드
if __name__ == "__main__":
    # 비디오 경로
    video_path = "C:/Users/wndhk/vision/bill_sample.mp4"  # 비디오 파일 경로 입력

    # 데이터 전처리
    print("Preprocessing video...")
    frames, coords = preprocess_video(video_path)
    print(f"Frames shape: {frames.shape}, Coords shape: {coords.shape}")

    # 모델 학습
    print("Training model...")
    trained_model = train_model(frames, coords)

    # 학습된 모델 저장
    save_model(trained_model, 'trained_model.pth')

    # 예측하기 위한 새로운 비디오 데이터
    new_video_path = "C:/Users/wndhk/vision/billl_sample2.mp4"  # 새로운 비디오 경로
    print("Preprocessing new video...")
    new_frames, _ = preprocess_video(new_video_path)
    predicted_trajectory = predict_trajectory(trained_model, new_frames)
    print(f"Predicted trajectory: {predicted_trajectory}")
