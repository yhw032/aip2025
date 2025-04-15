import torch

w = torch.tensor(2.0, requires_grad=False)

y = w ** 2
z = 2 * y + 5

print(z)