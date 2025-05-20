import torch
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

linear = nn.Linear(2, 1)
activation = nn.Sigmoid()
model = nn.Sequential(linear, activation).to(device)

criterion = nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=1)

num_epochs = 10000
for epoch in range(num_epochs + 1):
    hypothesis = model(X)
    cost = criterion(hypothesis, Y)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{num_epochs}: Cost = {cost.item()}")

with torch.no_grad():
    hypothesis = model(X)
    predicted = (hypothesis > 0.5).float()
    accuracy = (predicted == Y).float().mean()

    print(f"예측값: {predicted}, 정확도: {accuracy}")