import torch

w = torch.tensor(2.0, requires_grad=True)

n_epoch = 20
for epoch in range(n_epoch):
    z = 2 * w
    if w.grad is not None:
        w.grad.zero_()
    z.backward()
    print(w.grad)
