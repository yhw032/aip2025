import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import csv

# 1. 데이터셋 작성 ---------------------------
class HousePriceDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x = x_data
        self.y = y_data


    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

# 데이터셋 로드
x_origin = []
y_origin = []
with open('house_price_norm.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # 1행 헤더 스킵
    for row in reader:
        features = list(map(float, row[1:7]))
        label = float(row[7])  

        x_origin.append(features)
        y_origin.append([label])  

x_origin = torch.tensor(x_origin, dtype=torch.float32)
y_origin = torch.tensor(y_origin, dtype=torch.float32)

dataset = HousePriceDataset(x_origin, y_origin)


# Training Set, Test Set 분할 ------------
x_data = dataset.x
y_data = dataset.y

# 80%는 훈련용, 20%는 테스트용으로 분할
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, test_size=0.2, random_state=2)



train_set = HousePriceDataset(x_train, y_train)
test_set = HousePriceDataset(x_test, y_test)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False)


# 2. 모델 학습 (선형 회귀)

model = nn.Linear(6, 1)
optimizer = optim.SGD(model.parameters(), lr=1e-2)

num_epochs = 100
total_loss = []
# 학습 루프
for epoch in range(1, num_epochs + 1):
    for step, (x_batch, y_batch) in enumerate(train_loader):
        outputs = model(x_batch)  # 모델 예측
        loss = F.mse_loss(outputs, y_batch)  # 손실 계산

        optimizer.zero_grad()   # gradient 0으로 초기화
        loss.backward() # backward 연산
        optimizer.step()    # W, b 업데이트

        total_loss.append(loss.item())

    if epoch % 5 == 0:
        print(f"[Epoch {epoch}/{num_epochs}] Loss:{loss.item()}")


# Loss 시각화
plt.plot(range(len(total_loss)), total_loss)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss')
plt.show()


# 3. 모델 검증
with torch.no_grad():
    total_loss = 0.0
    acc = 0.0
    for x_test, y_test in test_loader:
        preds = model(x_test)   # 모델 예측값 preds에 저장
        loss = F.mse_loss(preds, y_test)  # 손실 계산
        total_loss += loss.item()       # 손실 누적
        
        acc = abs(preds - y_test).sum().item()
        
    # 평균 손실 계산
    avg_loss = total_loss / len(test_set)

    # 전체 정확도 계산
    avg_acc = acc / len(test_set)

    print(f"[*] error is {avg_loss}")
    print(f"[*] accuracy is {avg_acc}")
