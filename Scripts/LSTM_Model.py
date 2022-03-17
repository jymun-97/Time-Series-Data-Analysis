#%%
import numpy as np
from numpy.lib.twodim_base import _trilu_indices_form_dispatcher
import pandas as pd
from pandas.core.algorithms import mode
pd.options.display.float_format = '{:,.2f}'.format 

#torch related
import torch
from torch import nn
import torch.nn.functional as F

#pytorch lightning
import pytorch_lightning as pl

class LSTM(pl.LightningModule):
    def __init__(self, window, dim):
        super().__init__()
        self.window = window
        self.dim = dim
        self.hidden_size = 60
        self.dropout = 0.3
        self.layers = 2

        self.LSTM = nn.LSTM(
            input_size = self.dim, 
            hidden_size = self.hidden_size, 
            num_layers = self.layers, 
            batch_first = True, 
            bidirectional = False,
            dropout = self.dropout
        )
        self.LINEAR = nn.Linear(self.hidden_size, 24)

    def forward(self, x):                     # x        : (batch, 168, 24)
        lstm_out, _ = self.LSTM(x)            # lstm_out : (batch, 168, 60)
        target = lstm_out[:,-1]               # target   : (batch, 60)
        target = torch.unsqueeze(target, 1)   # target   : (batch, 1, 60)
        y_pred = self.LINEAR(target)          # y_pred   : (batch, 1, 24)
        y_pred = y_pred.view([-1, 24, 1])     # y_pred   : (batch, 24, 1)
        
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
        