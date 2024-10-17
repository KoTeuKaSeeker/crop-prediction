import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List

class CropDataset(Dataset):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, parameter_names: List[str]):
        self.x = x
        self.y = y
        self.context_size = self.x[0].shape[0]
        self.parameter_names = parameter_names

        self.mean = None
        self.std = None


    @classmethod
    def get_train_and_valid(cls, path, context_size, num_aug_copies=1, validation_split=0.2, count_date_intervals=3):
        df = pd.read_csv(path)

        date = pd.to_datetime(df['date'])
        hour = date.apply(lambda x: (x - pd.Timestamp(f"{x.year}-01-01")).total_seconds() / 3600)

        max_hours = 366 * 24
        size_interval = max_hours / count_date_intervals
        df[[f"hour_{i}" for i in range(count_date_intervals)]] = np.clip(hour.values[:, None].repeat(count_date_intervals, 1)/size_interval - np.arange(count_date_intervals)[None].repeat(len(hour), 0), 0.0, 1.0)

        df.drop(columns=['date'], inplace=True)

        parameter_names = df.columns

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

        train_dataset = cls(x[:num_train], y[:num_train], parameter_names)
        val_dataset = cls(x[num_train:], y[num_train:], parameter_names)

        train_dataset.mean, val_dataset.mean = mean, mean
        train_dataset.std, val_dataset.std = std, std

        return train_dataset, val_dataset

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)