import torch

data = [[1.0, 2.0], [3.0, 4.0]]     #float32

t = torch.tensor(data, device='cpu')
print(t)

#t = torch.tensor(data, device='cuda')
#t_gpu = t.to('cuda')


t = t.view(4, 1)
print(t)

t - t.transpose(0, 1)
print(t, t.shape)
