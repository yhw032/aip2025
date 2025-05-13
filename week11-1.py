import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

x_train = [[1, 2, 1, 1],
            [2, 1, 3, 2],
            [3, 1, 3, 4],
            [4, 1, 5, 5],
            [1, 7, 5, 5],
            [1, 2, 5, 6],
            [1, 6, 6, 6],
            [1, 7, 7, 7]]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]


x_train = torch.tensor(x_train, dtype=torch.float32)  # Convert to float32
y_train = torch.tensor(y_train, dtype=torch.long)  # Convert to long for classification

model = nn.Linear(4, 3)  # 4 input features, 3 output classes

optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 1000
for epoch in range(epochs + 1):
    prediction = model(x_train)
    cost = F.cross_entropy(prediction, y_train) # softmax + NLLLoss

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print(f'Epoch {epoch}/{epochs}, Cost: {cost.item()}')

new_var = torch.FloatTensor([[4,1,5,5]])
new_pred = F.softmax(model(new_var))
res_pred = torch.argmax(new_pred, 1)

print(new_pred)
print(res_pred)