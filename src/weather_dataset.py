import torch
from torch.utils.data import Dataset
import pandas as pd

class WeatherDataset(Dataset):
    def __init__(self, path, device): # Специально передаётся девайс, так как данных мало и их всех можно разместить в GPU
        super().__init__()
        df = pd.read_csv(path)
        df = df[['Temperature (C)', 'Humidity']] # Каждый час
        self.data = torch.tensor(df.values, dtype=torch.float32, device=device)
    
    def __getitem__(self, index):
        return self.data[index]
