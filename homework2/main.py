import torch
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WineDataset(Dataset):
  # TODO: 작성 필요
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.x = torch.tensor(self.data.iloc[:, :-1].values, dtype=torch.float32)
        self.y = torch.tensor(self.data.iloc[:, -1].values, dtype=torch.float32).view(-1, 1)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
dataset = WineDataset('winequality-red-rev.csv')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x_batch, y_batch = next(iter(dataloader))
print(x_batch.shape)
print(y_batch.shape)