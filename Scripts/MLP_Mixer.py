#%%
import numpy as np
import pandas as pd
from torch.nn.modules.dropout import Dropout
pd.options.display.float_format = '{:,.2f}'.format 

#torch related
import torch
from torch import nn
import torch.nn.functional as F

#pytorch lightning
import pytorch_lightning as pl

from einops import rearrange
from einops.layers.torch import Rearrange, Reduce

# %%
class MLP(nn.Module):
    def __init__(self, dim, hidden_size):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, dim)
        )
    
    def forward(self, x):
        return self.mlp(x)

class MixerLayer(nn.Module):
    def __init__(self, dim, window):
        super().__init__()

        self.token_mix = nn.Sequential(
            Rearrange('b n d -> b d n'), # (batch, 168, 24) -> (batch, 24, 168)
            MLP(window, hidden_size=130),#             output: (batch, 24, 168)
            Rearrange('b d n -> b n d')  # (batch, 24, 168) -> (batch, 168, 24)
        )
        self.channel_mix = nn.Sequential(
            MLP(dim, hidden_size=130)
        )
        
    def forward(self, x): 
        x_hat = self.token_mix(x)
        x_hat_prime = self.channel_mix(x_hat)
        x = x + x_hat_prime
        
        return x


# %%
class MLPMixer(pl.LightningModule):
    def __init__(self, window, dim):
        super().__init__()

        nLayers = 3
        self.mixer_layers = nn.ModuleList([])
        for _ in range(nLayers):
            self.mixer_layers.append(
                MixerLayer(dim, window)
            )

        self.mlp_head = nn.Linear(window * dim, 24)

    def forward(self, x):
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        
        x = x.reshape([-1, x.shape[1] * x.shape[2]])
        x = torch.unsqueeze(x, 1)
        out = self.mlp_head(x)
        y_pred = out.view([-1, 24, 1])

        return y_pred

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

# %%

# %%
