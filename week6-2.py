import torch
import torch.nn as nn
import torch.nn.functional as F


x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = nn.Linear(3, 1)

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

for epoch in range(2000):
    pred = model(x_train)
    cost = F.mse_loss(pred, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    # if epoch % 100 == 0:
    #     print(f'Epoch: {epoch}, Loss: {cost.item()}')

print(model.weight, model.bias)

new_data = torch.FloatTensor([[20, 210, 21]])
new_pred = model(new_data)

print(new_pred.item())