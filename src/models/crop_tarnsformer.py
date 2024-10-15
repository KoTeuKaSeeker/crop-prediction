import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass


@dataclass
class CropTransformerConfig():
    input_dim = 10
    context_size = 1024
    d_model = 512          # The number of expected features in the encoder/decoder inputs
    nhead = 8              # The number of heads in the multiheadattention models
    num_layers = 6  # The number of encoder layers in the encoder
    dim_feedforward = 2048  # The dimension of the feedforward network model
    dropout = 0.1          # Dropout value
    output_dim = 10


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
        self.input_scaler = Scaler(torch.zeros((1, 1, config.input_dim)), torch.ones((1, 1, config.input_dim)))
        self.input_linear = nn.Linear(config.input_dim, config.d_model)
        self.pos_encoder = nn.Embedding(config.context_size, config.d_model)

        encoder_layer = nn.TransformerEncoderLayer(config.d_model, config.nhead, config.dim_feedforward, config.dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        self.fc_out = nn.Linear(config.d_model, config.output_dim)

    def forward(self, x): # B, 1024, 7
        B, T, C = x.shape
        
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