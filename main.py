import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image

from tqdm import tqdm

import wandb


# =========================================================
# 1. 데이터 경로
# =========================================================

DATA_DIR = Path("Data")

TRAIN_IMAGE_DIR = DATA_DIR / "training" / "Train_image"
TRAIN_LABEL_DIR = DATA_DIR / "Training" / "Train_Label"

VALID_IMAGE_DIR = DATA_DIR / "validation" / "Valid_image"
VALID_LABEL_DIR = DATA_DIR / "validation" / "Valid_Label"


# =========================================================
# 2. 이미지 파일 목록
# =========================================================

train_images = list(TRAIN_IMAGE_DIR.rglob("*.png"))
valid_images = list(VALID_IMAGE_DIR.rglob("*.png"))


# =========================================================
# 3. Train 데이터에서 클래스 확인
# =========================================================

diagnosis_names = []

for image_file in train_images:

    class_folder = image_file.parent.name

    label_folder = TRAIN_LABEL_DIR / class_folder.replace("TS_", "TL_")

    label_file = label_folder / (image_file.stem + ".json")

    with open(label_file, "r", encoding="utf-8") as f:
        label_data = json.load(f)

    diagnosis_name = label_data["annotations"][0]["diagnosis_info"]["diagnosis_name"]

    diagnosis_names.append(diagnosis_name)


# =========================================================
# 4. 클래스 → 숫자 Label
# =========================================================

classes = sorted(set(diagnosis_names))

class_to_idx = {
    class_name: idx
    for idx, class_name in enumerate(classes)
}


# =========================================================
# 5. 이미지 전처리
# =========================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# =========================================================
# 6. Dataset
# =========================================================

class SkinDataset(Dataset):

    def __init__(
        self,
        image_files,
        label_dir,
        class_to_idx,
        transform=None,
        image_prefix=None,
        label_prefix=None
    ):

        self.image_files = image_files
        self.label_dir = label_dir
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.image_prefix = image_prefix
        self.label_prefix = label_prefix


    def __len__(self):

        return len(self.image_files)


    def __getitem__(self, idx):

        # 이미지 경로
        image_file = self.image_files[idx]

        # 이미지 폴더명
        class_folder = image_file.parent.name

        # 이미지 폴더 TS_ → 라벨 폴더 TL_
        label_folder = self.label_dir / class_folder.replace(
            self.image_prefix,
            self.label_prefix
        )

        # JSON 경로
        label_file = label_folder / (image_file.stem + ".json")

        # JSON 읽기
        with open(label_file, "r", encoding="utf-8") as f:
            label_data = json.load(f)

        # 진단명 추출
        diagnosis_name = label_data["annotations"][0]["diagnosis_info"]["diagnosis_name"]

        # 진단명 → 숫자 Label
        label = self.class_to_idx[diagnosis_name]

        # 이미지 읽기
        image = Image.open(image_file).convert("RGB")

        # 이미지 전처리
        if self.transform:
            image = self.transform(image)

        return image, label


# =========================================================
# 7. Dataset 생성
# =========================================================

train_dataset = SkinDataset(
    image_files=train_images,
    label_dir=TRAIN_LABEL_DIR,
    class_to_idx=class_to_idx,
    transform=transform,
    image_prefix="TS_",
    label_prefix="TL_"
)

valid_dataset = SkinDataset(
    image_files=valid_images,
    label_dir=VALID_LABEL_DIR,
    class_to_idx=class_to_idx,
    transform=transform,
    image_prefix="VS_",
    label_prefix="VL_"
)


# =========================================================
# 8. DataLoader
# =========================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=32,
    shuffle=False
)


# =========================================================
# 9. Device 설정
# =========================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("사용 장치:", device)


# =========================================================
# 10. ResNet18
# =========================================================

model = models.resnet18(weights="DEFAULT")

model.fc = nn.Linear(
    model.fc.in_features,
    len(classes)
)

model = model.to(device)


# =========================================================
# 11. Loss Function
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# 12. Optimizer
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

wandb.init(
    project="skin-disease-resnet18",
    name="resnet18-baseline",
    config={
        "model": "ResNet18",
        "batch_size": 32,
        "learning_rate": 0.001,
        "epochs": 10,
        "num_classes": len(classes)
    }
)

# =========================================================
# 13. 학습 설정
# =========================================================

num_epochs = 10

train_losses = []
valid_losses = []

train_accuracies = []
valid_accuracies = []

best_valid_accuracy = 0.0


# =========================================================
# 14. Training
# =========================================================

for epoch in range(num_epochs):

    model.train()

    train_total_loss = 0
    train_correct = 0
    train_total = 0


    for images, labels in tqdm(
        train_loader,
        desc=f"Epoch [{epoch + 1}/{num_epochs}] Train"
    ):

        # GPU/CPU로 이동
        images = images.to(device)
        labels = labels.to(device)

        # Gradient 초기화
        optimizer.zero_grad()

        # Forward
        outputs = model(images)

        # Loss 계산
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # 가중치 업데이트
        optimizer.step()


        # Loss 누적
        train_total_loss += loss.item()


        # Accuracy 계산
        predictions = outputs.argmax(dim=1)

        train_correct += (predictions == labels).sum().item()
        train_total += labels.size(0)


    # Epoch 평균
    train_average_loss = train_total_loss / len(train_loader)

    train_accuracy = train_correct / train_total


    train_losses.append(train_average_loss)
    train_accuracies.append(train_accuracy)


    # =====================================================
    # 15. Validation
    # =====================================================

    model.eval()

    valid_total_loss = 0
    valid_correct = 0
    valid_total = 0


    with torch.no_grad():

        for images, labels in tqdm(
            valid_loader,
            desc=f"Epoch [{epoch + 1}/{num_epochs}] Valid"
        ):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            valid_total_loss += loss.item()


            predictions = outputs.argmax(dim=1)

            valid_correct += (predictions == labels).sum().item()
            valid_total += labels.size(0)


    # Validation 평균
    valid_average_loss = valid_total_loss / len(valid_loader)

    valid_accuracy = valid_correct / valid_total


    valid_losses.append(valid_average_loss)
    valid_accuracies.append(valid_accuracy)


    # =====================================================
    # 16. W&B 기록
    # =====================================================

    wandb.log({
        "train_loss": train_average_loss,
        "train_accuracy": train_accuracy,
        "valid_loss": valid_average_loss,
        "valid_accuracy": valid_accuracy
    })


    # =====================================================
    # 17. Best Model 저장
    # =====================================================

    if valid_accuracy > best_valid_accuracy:

        best_valid_accuracy = valid_accuracy

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "valid_accuracy": valid_accuracy,
            "valid_loss": valid_average_loss,
            "classes": classes
        }, "best_resnet18.pth")

        wandb.log({
            "best_valid_accuracy": best_valid_accuracy
        })

        print(
            f"Best Model 저장 완료 | "
            f"Epoch: {epoch + 1} | "
            f"Valid Accuracy: {valid_accuracy:.4f}"
        )


    # =====================================================
    # 18. Epoch 결과 출력
    # =====================================================

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_average_loss:.4f} "
        f"Train Accuracy: {train_accuracy:.4f} "
        f"Valid Loss: {valid_average_loss:.4f} "
        f"Valid Accuracy: {valid_accuracy:.4f}"
    )

wandb.finish()