import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


@dataclass
class CropTransformerConfig():
    input_dim = 11
    d_input_linear = 2048
    context_size = 1024
    d_model = 768          # The number of expected features in the encoder/decoder inputs
    nhead = 12             # The number of heads in the multiheadattention models
    num_layers = 10  # The number of encoder layers in the encoder
    dim_feedforward = 2048  # The dimension of the feedforward network model
    dropout = 0.1          # Dropout value
    output_dim = 11
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
    def __init__(self, config: CropTransformerConfig):
        super().__init__()
        self.config = config

        self.input_scaler = Scaler(torch.zeros((1, 1, config.input_dim)), torch.ones((1, 1, config.input_dim)))
        self.input_proj = nn.Linear(config.input_dim, config.d_input_linear)
        self.input_linear = nn.Linear(config.d_input_linear, config.d_model)
        self.pos_encoder = nn.Embedding(config.context_size, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.fc_out = nn.Linear(config.d_model, config.output_dim)
        
        self.count_date_intervals = nn.Parameter(torch.tensor(config.count_date_intervals, dtype=torch.int32), requires_grad=False)

    def forward(self, x): # B, 1024, 7
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
        x = self.input_scaler(x)
        x = self.forward(x)
        x = self.input_scaler.inv_forward(x)
        return x
    

    def generate(self, x, count_generations=1, context_size=-1):
        B, T, C = x.shape

        if context_size < 0:
            context_size = self.pos_encoder.num_embeddings
        
        count_date_intervals = self.count_date_intervals.item()
        size_interval = 366 * 24 / count_date_intervals
        last_date_hours = torch.sum(x[:, -1, -count_date_intervals:], dim=1)[:, None] * size_interval # B, 1
        generated_hours = torch.arange(1, count_generations+1, device=x.device)[None].repeat(B, 1) + last_date_hours
        generated_date = torch.clip(generated_hours[:, :, None].repeat(1, 1, count_date_intervals)/size_interval - torch.arange(count_date_intervals, device=x.device)[None, None, :].repeat(B, count_generations, 1), 0.0, 1.0)

        with torch.no_grad():
            idx = x
            for step in range(count_generations):
                idx_slice = idx if idx.size(1) < context_size else idx[:, -context_size:]
                pred = self.sfroward(idx_slice) # B, T, output_dim
                last_pred = pred[:, -1, :] # B, output_dim
                last_pred[:, -count_date_intervals:] = generated_date[:, step]
                
                idx = torch.cat((idx, last_pred[:, None, :]), dim=1)
        
        return idx[:, -count_generations:]
            

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
    
    def save_checkpoint(self, filename: str, optimizer: torch.optim.Optimizer, val_loss):
        """
        Сохраняет только один чекпоинт
        """
        checkpoint = {
            'model': self.state_dict(),
            'model_config': self.config,
            'optimizer': optimizer.state_dict(),
            'train_config': self.config,
            'metrics': {
                'val_loss': val_loss
            }
        }
        torch.save(checkpoint, filename)
    
    @classmethod
    def from_checkpoint(cls, path: str):
        """
        Loads the model from the specified path.
        """

        checkpoint = torch.load(path)

        model = cls(checkpoint['config'])
        optimizer = torch.optim.Optimizer()

        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        val_loss = checkpoint['val_loss']

        return model, optimizer, val_loss