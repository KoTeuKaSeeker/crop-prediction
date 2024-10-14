from src.models.crop_tarnsformer import CropTransformer, CropTransformerConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import numpy as np


class CropDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.context_size = self.x[0].shape[0]

    @classmethod
    def get_train_and_valid(cls, path, context_size, num_aug_copies=1, validation_split=0.2):
        df = pd.read_csv(path)

        df['date'] = pd.to_datetime(df['date'])

        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour

        df.drop(columns=['date'], inplace=True)

        mean = df.mean()
        std = df.std()

        df = (df - mean) / std

        in_num_batches = len(df) // context_size
        offset_size = context_size // num_aug_copies
        data_tensor = torch.tensor(df.values, dtype=torch.float32)

        x, y = [], []
        for idx in range(num_aug_copies):
            offset = idx * offset_size
            x.append((data_tensor[offset:-1-context_size+offset])[:-(len(df)%context_size)+1].view(in_num_batches-1, -1, data_tensor.shape[-1]))
            y.append((data_tensor[offset+1:-context_size+offset])[:-(len(df)%context_size)+1].view(in_num_batches-1, -1, data_tensor.shape[-1]))
        x, y = torch.cat(x), torch.cat(y)

        perm = torch.randperm(len(x))
        x, y = x[perm], y[perm]
        
        num_train = int((1 - validation_split) * len(x))

        train_dataset = cls(x[:num_train], y[:num_train])
        val_dataset = cls(x[num_train:], y[num_train:])

        return train_dataset, val_dataset

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    context_size = 128
    batch_size = 16
    epochs = 1000
    learning_rate = 1e-6

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset, val_dataset = CropDataset.get_train_and_valid("data/argo_dataset/argo_dataset.csv", 
                                                                 context_size=context_size, 
                                                                 num_aug_copies=5)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_dataset = DataLoader(val_dataset, batch_size, shuffle=True)

    config = CropTransformerConfig()
    model = CropTransformer(config)
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    total_step = 0
    for epoch in range(epochs):
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            print(f"total_step {total_step} | loss: {loss.item()}")
            
            total_step += 1

    