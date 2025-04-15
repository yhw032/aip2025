import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset

class MyDataset(Dataset):
    def __init__(self, x_data, y_data):
        self.x_data = x_data
        self.y_data = y_data

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.y_data[idx]


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = MyDataset(x_train, y_train)
dataloader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
# shuffle=True: 데이터셋을 섞어서 배치로 나누기
# drop_last=True: 마지막 배치가 batch_size보다 작을 경우 버리기
# drop_last=False: 마지막 배치가 batch_size보다 작을 경우 랜덤으로 패딩하기(기본값)

model = nn.Linear(3, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20

for epoch in range(nb_epochs + 1):
    for i, samples in dataloader:
        x_train, y_train = samples

        pred = model(x_train)
        cost = F.mse_loss(pred, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {.4d}/{} Cost: {:.6f}'.format(epoch, nb_epochs, cost.item()))

print(model.weight, model.bias)

new_data = torch.FloatTensor([[20, 210, 21]])
new_pred = model(new_data)

print(new_pred.item())