#%%
import numpy as np
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format 

#torch related
import torch
from torch import nn
import torch.nn.functional as F

#pytorch lightning
import pytorch_lightning as pl

class FNN(pl.LightningModule):
    def __init__(self, window, dim):
        super().__init__()
        self.window = window
        self.dim = dim
        self.dropout = 0.05

        self.FNN = nn.Sequential(
            nn.Linear(self.dim * self.window, 1024),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(64, 24),
        )

    def forward(self, x):                            # x   : (batch, 168, 24)
        x = x.reshape([-1, x.shape[1] * x.shape[2]]) # x   : (batch, 168*24)
        x = torch.unsqueeze(x, 1)                    # x   : (batch, 1, 168*24)
        out = self.FNN(x)                            # out : (batch, 1, 24)
        y_pred = out.view([-1, 24, 1])               # out : (batch, 24, 1)

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
        