import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import yaml
import os
import pandas as pd
import numpy as np


@dataclass
class CropTransformerConfig():
    input_dim = 14
    d_input_linear = 2048
    context_size = 1024
    d_model = 768          # The number of expected features in the encoder/decoder inputs
    nhead = 12             # The number of heads in the multiheadattention models
    num_layers = 10  # The number of encoder layers in the encoder
    dim_feedforward = 2048  # The dimension of the feedforward network model
    dropout = 0.1          # Dropout value
    output_dim = 14
    count_date_intervals = 4


class Scaler(nn.Module):
    def __init__(self, mean, std):
        super().__init__()
        self.mean = nn.Parameter(mean, requires_grad=False)
        self.std = nn.Parameter(std, requires_grad=False)
    
    def forward(self, x):
        return (x - self.mean) / self.std
    
    def inv_forward(self, x):
        return x * self.std + self.mean


class CropTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.input_scaler = Scaler(torch.zeros((1, 1, self.config["input_dim"])), torch.ones((1, 1, self.config["input_dim"])))
        self.input_proj = nn.Linear(self.config["input_dim"], self.config["d_input_linear"])
        self.input_linear = nn.Linear(self.config["d_input_linear"], self.config["d_model"])
        self.pos_encoder = nn.Embedding(self.config["context_size"], self.config["d_model"])

        encoder_layer = nn.TransformerEncoderLayer(self.config["d_model"], self.config["nhead"], self.config["dim_feedforward"], self.config["dropout"])
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config["num_layers"])

        self.fc_out = nn.Linear(self.config["d_model"], self.config["output_dim"])
        
        self.config["count_date_intervals"] = nn.Parameter(torch.tensor(self.config["count_date_intervals"], dtype=torch.int32), requires_grad=False)
    
    @classmethod
    def from_config(cls, path: str):
        with open(path, 'r') as file:
            config = yaml.safe_load(file)
        model = cls(config)
        return model

    def forward(self, x):
        """
            Предсказывает следующее значение для каждого параметра временного ряда. 
            Параметры имеют следующий вид:
            [SOLAR_RADIATIO,
            PRECIPITATION,
            WIND_SPEED,
            LEAF_WETNESS,
            HC_AIR_TEMPERATURE,
            HC_RELATIVE_HUMIDITY,
            DEW_POINT,
            hour,
            day,
            month,
            hour_interval_1,
            hour_interval_2,
            ...
            hour_interval_<count_date_intervals>]
            
            **Принимает на вход нормализированные данные**
        """
        B, T, C = x.shape
        
        x = nn.ReLU()(self.input_proj(x)) # Этот линейный слой нужен, чтобы учесть большую дискретность входных значений
        x = self.input_linear(x) # B, T, d_model        
        pos_emb = self.pos_encoder(torch.arange(0, T, device=x.device)) # T, d_model
        x = x + pos_emb # B, T, d_model

        mask = torch.nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        x = self.transformer_encoder(x, mask=mask, is_causal=True) # B, T, d_model

        x = self.fc_out(x) # B, T, output_dim

        return x

    def sfroward(self, x):
        """
            Более удобная функция, дла предсказания значений. Предсказывает следующее значение 
            для каждого параметра временного ряда, при этом **принимает на вход НЕ нормализированные данные**
            Параметры имеют следующий вид:
            [SOLAR_RADIATIO,
            PRECIPITATION,
            WIND_SPEED,
            LEAF_WETNESS,
            HC_AIR_TEMPERATURE,
            HC_RELATIVE_HUMIDITY,
            DEW_POINT,
            hour,
            day,
            month,
            hour_interval_1,
            hour_interval_2,
            ...
            hour_interval_<count_date_intervals>]
        """

        x = self.input_scaler(x)
        x = self.forward(x)
        x = self.input_scaler.inv_forward(x)
        return x
    

    def generate(self, x, count_generations=1, context_size=-1):
        """
            Предсказывает **count_generations** следующий значений временного ряда. Принимает на
            вход **не нормализированные данные**.
            Параметры имеют следующий вид:
            [SOLAR_RADIATIO,
            PRECIPITATION,
            WIND_SPEED,
            LEAF_WETNESS,
            HC_AIR_TEMPERATURE,
            HC_RELATIVE_HUMIDITY,
            DEW_POINT,
            hour,
            day,
            month,
            hour_interval_1,
            hour_interval_2,
            ...
            hour_interval_<count_date_intervals>]
        """

        B, T, C = x.shape

        if context_size < 0:
            context_size = self.pos_encoder.num_embeddings
        
        count_date_intervals = self.config["count_date_intervals"].item()
        size_interval = 366 * 24 / count_date_intervals
        last_date_hours = torch.sum(x[:, -1, -count_date_intervals:], dim=1)[:, None] * size_interval # B, 1
        generated_hours = torch.arange(1, count_generations+1, device=x.device)[None].repeat(B, 1) + last_date_hours
        generated_date = torch.clip(generated_hours[:, :, None].repeat(1, 1, count_date_intervals)/size_interval - torch.arange(count_date_intervals, device=x.device)[None, None, :].repeat(B, count_generations, 1), 0.0, 1.0)

        dates = x[:, -1:, -count_date_intervals-3:-count_date_intervals].repeat(1, count_generations, 1).cpu() # B, count_generations, 3
        id = 1
        for b in range(B):
            for t in range(count_generations):
                date = pd.Timestamp(hour=int(dates[b, t, 0].item()), 
                                    day=int(dates[b, t, 1].item()), 
                                    month=int(dates[b, t, 2].item()), year=2024)
                new_date = date + pd.Timedelta(id, unit='h')
                dates[b, t, :] = torch.tensor([new_date.hour, new_date.day, new_date.month])
                id += 1
        
        dates = dates.to(x.device)

        with torch.no_grad():
            idx = x
            for step in range(count_generations):
                idx_slice = idx if idx.size(1) < context_size else idx[:, -context_size:]
                pred = self.sfroward(idx_slice) # B, T, output_dim
                last_pred = pred[:, -1, :] # B, output_dim
                last_pred[:, -count_date_intervals:] = generated_date[:, step]
                last_pred[:, -count_date_intervals-3:-count_date_intervals] = dates[:, step]
                
                idx = torch.cat((idx, last_pred[:, None, :]), dim=1)
        
        return idx[:, -count_generations:]
    

    def predict(self, x: np.ndarray, count_generations=1, context_size=-1):
        """
            Более удобная версия функции generate. Предсказывает **count_generations** следующий 
            значений временного ряда. Принимает на вход NumPy массив размером (T, C), где
            T - количество временных шагов, C - количество входных параметров метеостанции. Все данные 
            подаются в естественном, не нормализованном виде для удобства использования.
            Параметры метеостанции, которые должны содержаться в x:
            [SOLAR_RADIATIO,
            PRECIPITATION,
            WIND_SPEED,
            LEAF_WETNESS,
            HC_AIR_TEMPERATURE,
            HC_RELATIVE_HUMIDITY,
            DEW_POINT,
            hour,
            day,
            month]
        """
        T, C = x.shape
        
        count_date_intervals = self.config["count_date_intervals"].item()
        max_hours = 366 * 24
        size_interval = max_hours / count_date_intervals
        hour_interval = np.zeros((T, count_date_intervals))
        for t in range(T):
            hour = (pd.Timestamp(hour=int(x[t, -3]), day=int(x[t, -2]), month=int(x[t, -1]), year=2024) - pd.Timestamp("2024-01-01")).total_seconds() / 3600
            hour_interval[t, :] = np.clip(np.array([hour]*count_date_intervals)/size_interval - np.arange(count_date_intervals), 0.0, 1.0)
        
        x = np.concatenate((x, hour_interval), axis=1)
        x = torch.tensor(x, dtype=torch.float32, device=self.input_linear.weight.device)[None]

        pred = self.generate(x, count_generations, context_size)

        return pred[0].cpu().numpy()


    def init_scaler(self, mean, std):
        self.input_scaler = Scaler(mean, std)
        return self
    

    def configure_optimizers(self, weight_decay, learning_rate):
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for pn, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate)
        return optimizer
    
    def save_checkpoint(self, filename: str, optimizer: torch.optim.Optimizer, train_config: dict, val_loss):
        """
        Save a checkpoint.
        """
        checkpoint = {
            'model': self.state_dict(),
            'model_config': self.config,
            'optimizer': optimizer.state_dict(),
            'train_config': train_config,
            'metrics': {
                'val_loss': val_loss
            }
        }
        torch.save(checkpoint, filename)
    
    def save(self, dir_path: str, optimizer: torch.optim.Optimizer, train_config: dict, metrics: dict, best_metrics: dict):
        """
        Save the last and the best versions of the model.
        """

        last_model_path = os.path.join(dir_path, "last.pt")
        self.save_checkpoint(last_model_path, optimizer, train_config, metrics["val_loss"])

        if best_metrics["val_loss"] is None or metrics["val_loss"] <= best_metrics["val_loss"]:
            best_metrics["val_loss"] = metrics["val_loss"]
            best_model_path = os.path.join(dir_path, "best.pt")
            self.save_checkpoint(best_model_path, optimizer, train_config, metrics["val_loss"])
    
    @classmethod
    def from_checkpoint(cls, path: str, device):
        checkpoint = torch.load(path)

        model = cls(checkpoint['model_config'])
        model.load_state_dict(checkpoint["model"])
        model.to(device)

        train_config = checkpoint["train_config"]
        # if train_config["optimizer_type"] == 'AdamW':
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(train_config['learning_rate']))

        optimizer = model.configure_optimizers(0.1, float(train_config['learning_rate']))
        optimizer.load_state_dict(checkpoint["optimizer"])

        return model, optimizer, train_config, checkpoint['metrics']