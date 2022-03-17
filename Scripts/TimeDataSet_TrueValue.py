#%%
import torch
import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format 
import numpy as np
import os, glob, json, pickle
from torch.utils.data import DataLoader, random_split
from sklearn.model_selection import train_test_split

class TimeseriesPredictionDataset_TrueValue(torch.utils.data.Dataset):
    def __init__(self, TARGET_DATA, samprate, window, stride, json_root, scaler = 'MMSC', fileformat = 'pkl'):
        self.TARGET_DATA = TARGET_DATA
        with open(json_root, 'rb') as f:
            DataInfo = json.load(f)
        self._ROOT_ = DataInfo['ROOT'] + TARGET_DATA
        self.TRJ_CLASS = DataInfo[TARGET_DATA]['TRJ_CLASS']
        self.WINDOW = window
        self.SAMPRATE = samprate
        self.STRIDE = stride
        self.FFORMAT = fileformat

        self.file_list = []
        for i in self.TRJ_CLASS:
            self.file_list += glob.glob(self._ROOT_ + f'/{i}/*.{self.FFORMAT}')

        self.X_DATA = [] 
        self.Y_DATA = []

        with open(self.file_list[0], 'rb') as f:
            temp = pickle.load(f) 

        # temp = self.SCALER.transform(temp)    
        self.len, self.dim = temp.shape
        while (i + self.WINDOW < self.len):
            sample_x = temp[i : i + self.WINDOW : self.SAMPRATE]
            sample_y = temp[i + self.WINDOW : i + self.WINDOW + 24 : self.SAMPRATE]
            
            self.X_DATA.append(sample_x)
            self.Y_DATA.append(sample_y)
            i += self.STRIDE

    def __len__(self):
        return (len(self.X_DATA))

    def __getitem__(self, idx):
        return [self.X_DATA[idx], self.Y_DATA[idx]]

# %%
