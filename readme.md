Skin Disease Classification with ResNet18 & W&B
PyTorch와 사전 학습된(Pretrained) ResNet18 모델을 활용하여 피부 질환 이미지를 학습하고 분류하는 파이프라인 프로젝트입니다. Weights & Biases(W&B)를 연동하여 학습 과정에서의 Loss와 Accuracy를 실시간으로 모니터링할 수 있습니다.


🚀 주요 기능 (Features)
Custom Dataset: 폴더 구조 기반의 이미지 및 JSON 라벨 데이터 로딩 (SkinDataset)

Transfer Learning: PyTorch의 사전 학습된 ResNet18 백본 활용 및 클래스 개수에 맞춘 Fully Connected Layer 커스텀

Experiment Tracking: Weights & Biases(W&B)를 통한 손실(Loss) 및 정확도(Accuracy) 트래킹

Model Checkpoint: 검증 정확도(Validation Accuracy) 기준 최고 성능의 모델(best_resnet18.pth) 자동 저장


📂 프로젝트 구조 (Project Structure)
Plaintext
├── Data/
│   ├── training/
│   │   └── Train_image/
│   ├── Training/
│   │   └── Train_Label/
│   ├── validation/
│   │   ├── Valid_image/
│   │   └── Valid_Label/
├── main.py
├── requirements.txt
└── README.md


⚙️ 설치 방법 (Installation)
저장소를 클론하거나 프로젝트 폴더로 이동합니다.

필수 라이브러리를 설치합니다.

Bash
pip install -r requirements.txt


🏃‍♂️ 사용 방법 (Usage)
Data 폴더 내에 학습 및 검증 이미지와 라벨 데이터가 올바른 경로에 위치해 있는지 확인합니다.

터미널에서 아래 명령어를 실행하여 학습을 시작합니다.

Bash
python main.py
W&B 로그인 상태에서 프로젝트 대시보드에 접속하여 실시간 학습 지표를 확인합니다.