from src.models.crop_tarnsformer import CropTransformer, CropTransformerConfig
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import re
import numpy as np
import time


class CropDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor):
        self.x = x
        self.y = y
        self.context_size = self.x[0].shape[0]

        self.mean = None
        self.std = None


    @classmethod
    def get_train_and_valid(cls, path, context_size, num_aug_copies=1, validation_split=0.2):
        df = pd.read_csv(path)

        df['date'] = pd.to_datetime(df['date'])

        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['hour'] = df['date'].dt.hour

        df.drop(columns=['date'], inplace=True)

        mean = torch.tensor(df.mean().values, dtype=torch.float32)[None, None, ...]
        std = torch.tensor(df.std().values, dtype=torch.float32)[None, None, ...]

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

        train_dataset.mean, val_dataset.mean = mean, mean
        train_dataset.std, val_dataset.std = std, std

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
    model = CropTransformer(config).init_scaler(train_dataset.mean, train_dataset.std)
    model = model.to(device)

    optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=learning_rate)
    criterion = nn.MSELoss()

    model.train()
    total_step = 0
    for epoch in range(epochs):
        t0 = time.time()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            x, y = model.input_scaler(x), model.input_scaler(y)
            
            optimizer.zero_grad()
            preds = model(x)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            
            torch.cuda.synchronize()
            t1 = time.time()
            dt = t1 - t0
            print(f"total_step {total_step} | loss: {loss.item()} | dt: {dt:.3f}")
            
            total_step += 1
            t0 = time.time()