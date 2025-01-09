# Image_processing_Deep_learning

## 공 궤적 분석 
- 2D 이미지 upper,down RGB를 이용해서 중심점 인식 + 경로 표시
- 3D 이미지 크기 restrict 로 인식 + 움직이는 경로 표시 (저장된 동영상)


## 모델 포즈 분석
- mediapipe를 이용한 사용자 자세 분석
> 관절을 기준으로 각 포인트를 받아서 인식
- 시용시 실시간, 저장된 영상 분석 가능 + status
- 영상을 학습 시켜 model.pkt 생성 + 생성 된 모델을 호출 후 저장 영상 인식
