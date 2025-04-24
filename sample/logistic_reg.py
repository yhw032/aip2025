import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import csv

# Custom Dataset 정의
class WineDataset(Dataset):
    def __init__(self, csv_file):
        self.x = []
        self.y = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                features = list(map(float, row[:11]))
                quality = float(row[11])  # 회귀 문제이므로 float로 받음
                # quality 점수를 0 또는 1로 변환
                label = 1 if quality >= 6 else 0

                self.x.append(features)
                self.y.append([label])  # 로지스틱은 float로 [0.0] or [1.0]

        self.x = torch.tensor(self.x, dtype=torch.float32) # 실수형으로 변환
        self.y = torch.tensor(self.y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 데이터셋 로드 및 Train/Validation Split
dataset = WineDataset('winequality-red-rev.csv')

x_data = dataset.x
y_data = dataset.y

x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=42)

train_set = TensorDataset(x_train, y_train)
test_set = TensorDataset(x_test, y_test)

# train_size = int(len(dataset) * 0.8)        # 80% 훈련 데이터
# test_size = len(dataset) - train_size        # 20% 검증 데이터
# # 랜덤하게 분할
# train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)

# 디버그용 데이터셋 출력
# print("Train set size:", len(train_set))
# for x_batch, y_batch in train_loader:
#     print(f"x_batch shape: {x_batch}, y_batch shape: {y_batch}")


model = nn.Sequential(
    nn.Linear(11, 1),  # 입력층: 11개 특성, 은닉층: 64개 노드
    nn.Sigmoid(),  # 활성화 함수
)
optimizer = optim.SGD(model.parameters(), lr=5e-5)


num_epochs = 1000
# 학습 루프
for epoch in range(1, num_epochs + 1):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for step, (x_batch, y_batch) in enumerate(train_loader):
        outputs = model(x_batch)  # 모델 예측
        loss = F.binary_cross_entropy(outputs, y_batch)  # BCE 손실 함수 사용

        optimizer.zero_grad()   # gradient 0으로 초기화
        #outputs = model(x_batch)
        #loss = criterion(outputs, y_batch)
        loss.backward() # backward 연산
        optimizer.step()    # W, b 업데이트

        total_loss += loss.item() * x_batch.size(0)
        total_samples += x_batch.size(0)

    avg_loss = total_loss / total_samples

    # 검증 정확도 측정 (정수로 반올림한 예측 기준)
    if epoch % 50 == 0:
        model.eval()        # 모델 평가 모드 전환
        with torch.no_grad():       # gradient 계산 안함(메모리 절약)
            test_correct = 0
            test_total = 0
            for x_test, y_test in test_loader: 
                preds = model(x_test)   # 모델 예측값 preds에 저장
                preds_rounded = preds.round()   # 예측값을 정수로 반올림(preds는 실수형)
                test_correct += (preds_rounded == y_test).sum().item()  
                # 예측값과 실제값 비교하고 맞으면 1, 틀리면 0 저장
                # item()은 tensor를 python 숫자로 변환

                test_total += y_test.size(0)
            accuracy = test_correct / test_total * 100  # 맞은 개수 / 전체 개수 * 100

        print(f"[Epoch {epoch}/{num_epochs}] Loss:{loss.item()} , Avg Loss: {avg_loss:.4f}, Val Accuracy: {accuracy:.2f}%")



# 학습 종료 후 테스트 데이터 예측 결과 출력
model.eval()
with torch.no_grad():
    print("\n[Test Data Prediction Results]")
    for x_test, y_test in test_loader:
        preds = model(x_test)
        preds_rounded = preds.round()
        for i in range(len(x_test)):
            print(f"예측값: {preds[i].item():.2f} → 반올림: {preds_rounded[i].item():.0f} | 실제값: {y_test[i].item():.0f} | 오차: {abs(preds[i].item() - y_test[i].item()):.2f}")
