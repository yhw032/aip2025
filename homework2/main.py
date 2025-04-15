import torch
import numpy as np
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WineDataset(Dataset):
    def __init__(self, csv_file):
        self.x = []
        self.y = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)    # Skip header
            for row in reader:
                features = list(map(float, row[:11])) # First 11 columns are features
                label = int(row[11]) # Last column is label

                self.x.append(features)
                self.y.append(label)

        self.x = torch.tensor(self.x, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
dataset = WineDataset('winequality-red-rev.csv')
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
x_batch, y_batch = next(iter(dataloader))
print(x_batch.shape)
print(y_batch.shape)