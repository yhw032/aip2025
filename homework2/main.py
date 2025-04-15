import torch
import csv
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class WineDataset(Dataset):
    def __init__(self, csv_file):
        self.x = []
        self.y = []
        with open(csv_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)    # 첫 행 데이터 무시
            for row in reader:
                features = list(map(float, row[:11]))   # 1~11열 feature
                quality = int(row[11])                  # 12열 와인품질(quality)

                self.x.append(features)
                self.y.append(quality)

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