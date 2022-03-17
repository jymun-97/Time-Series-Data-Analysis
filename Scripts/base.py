#%%
### Import Libraries

# Ignore the warnings
from distutils import version
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

import json
import numpy as np
np.set_printoptions(formatter={'float_kind': lambda x: "{0:0.4f}".format(x)})
import pandas as pd
pd.options.display.float_format = '{:,.3f}'.format 

#torch related
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

#pytorch lightning
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.csv_logs import CSVLogger
from pytorch_lightning.callbacks import EarlyStopping

# import custom class for time series dataset
from TimeDataSet import TimeseriesPredictionDataset
from TimeDataSet_TrueValue import TimeseriesPredictionDataset_TrueValue

# others
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import os, json, pickle
import six

from LSTM_Model import LSTM
from FNN_Model import FNN
from MLP_Mixer import MLPMixer

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#%%
### Define Functions

# Create and Train Model
def create_model(model_name):
    MODELS = {'LSTM':LSTM, 'FNN':FNN, 'MLP_Mixer':MLPMixer}
    
    csv_logger = CSVLogger('../Log/', name=model_name+'_Log', version='0')
    MODEL = MODELS[model_name](window=WINDOW, dim=DIM)

    trainer = pl.Trainer(
        gpus=[1],
        max_epochs=MAX_EPOCHS, 
        logger=csv_logger,
    )
    
    trainer.fit(MODEL, train_loader, val_dataloaders=val_loader)
    trainer.test(test_dataloaders=test_loader)

    return MODEL

# Get Predict Values
def predict(model_name, index):
    MODELS = {'LSTM':lstm, 'FNN':fnn, 'MLP_Mixer':mlp_mixer}
    model = MODELS[model_name]

    temp = np.array(test)               # Test Set: (X, Y)
    temp = temp[:, 0]                   # X values of Test Set
    x = torch.unsqueeze(temp[index], 0) # x : (1, 168, 24)
    
    y_pred = model(x)                   # (1, 24, 1)
    y_pred = torch.squeeze(y_pred)      # (24, 1)
    y_pred = np.reshape(y_pred.detach().numpy(), (-1, 1)) # torch to numpy
    y_pred = y_scaler.inverse_transform(y_pred)           # to true value
    y_pred = np.reshape(y_pred, (-1, 1))
    
    return torch.Tensor(y_pred)

def visualize(model_name, start_index, steps):
    if start_index == 0: str_index = 'first'
    elif start_index + steps == len(test): str_index = 'last'
    else: str_index = str(start_index) + 'th'
    save_path = IMAGE_PATH + model_name + '/Predict_' + str_index + '_index.png'
    ensure_dir(save_path)

    result_df = pd.DataFrame()
    for i in range(start_index, start_index + steps):
        predict_df = pd.DataFrame(np.array(predict(model_name, i)))
        raw_df = test_tv[i][1]

        predict_df.index = raw_df.index
        df = pd.concat([raw_df['load'], predict_df], axis=1)
        result_df = pd.concat([result_df, df])
    
    result_df.columns = ['Raw Data', model_name]
    result_df.plot(figsize=(16, 5), legend=True, title=model_name + '(' + str_index + ' index)')

    plt.savefig(save_path)
    plt.show()

    return predict_df

# Get Loss Values
def analysis(model_name):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
    loss_df = pd.DataFrame()
        
    metrics = pd.read_csv('/home/mjy0750/Project/Log/' + model_name + '_Log/0/metrics.csv')
    train_loss = metrics[['train_loss', 'step', 'epoch']][~np.isnan(metrics['train_loss'])]
    val_loss = metrics[['val_loss', 'epoch']][~np.isnan(metrics['val_loss'])]
    test_loss = metrics['test_loss'].iloc[-1]

    axes[0].set_title('Train loss per epoch')
    axes[0].plot(train_loss['epoch'], train_loss['train_loss'])
    axes[1].set_title('Validation loss per epoch')
    axes[1].plot(val_loss['epoch'], val_loss['val_loss'], color='orange')

    train_loss = train_loss['train_loss'].iloc[-1]
    val_loss = val_loss['val_loss'].iloc[-1]
    test_loss = test_loss

    loss = [train_loss, val_loss, test_loss]
    loss = np.reshape(loss, (1, 3))
    loss = pd.DataFrame(loss)
    loss_df = pd.concat([loss_df, loss])

    plt.savefig(IMAGE_PATH + model_name + '/Loss Graph.png')
    plt.show()

    loss_df.columns = ['Train Loss', 'Val Loss', 'Test Loss']
    loss_df.index = [model_name]
    
    _, loss_table = render_mpl_table(loss_df, model_name)
    loss_table.savefig(IMAGE_PATH + model_name + '/Loss Values.png')

# Evaluate
def score(y_pred, y):
    MAE = F.l1_loss(y_pred, y)
    MAPE = torch.mean(torch.abs((y_pred - y) / y)) * 100
    MPE = torch.mean(((y_pred - y) / y)) * 100
    MSE = F.mse_loss(y_pred, y)
    RMSE = np.sqrt(MSE)
    RRMSE = RMSE / torch.mean(y) * 100 

    return MAE, MAPE, MPE, MSE, RMSE, RRMSE

def summarize(model_name):
    test_data = np.array(test) # Test Set: (X, Y)
    Y_test = test_data[:, 1]   # Y Values of Test Set 
    scores = []

    for i in range(len(Y_test)):
        Y_pred = predict(model_name, i)

        Y = Y_test[i]
        Y = np.reshape(Y.detach().numpy(), (-1, 1))
        Y = y_scaler.inverse_transform(Y)
        Y = torch.Tensor(Y)

        scores.append(score(Y_pred, Y))

    scores = np.array(scores).mean(axis=0)
    scores = np.reshape(scores, [1, -1])

    df_scores = pd.DataFrame(scores)
    df_scores.columns = ['MAE', 'MAPE', 'MPE', 'MSE', 'RMSE', 'RRMSE']
    df_scores.index = [model_name]
    
    _, score_table = render_mpl_table(df_scores, model_name)
    score_table.savefig(IMAGE_PATH + model_name + '/Score.png')

# DataFrame to Image
def render_mpl_table(data, model_name, col_width=2.0, row_height=0.625, font_size=14,
                     header_color='#A6A6A6', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    data.update(data[:].astype(float))
    data.update(data[:].applymap('{:,.3f}'.format))
    df = pd.DataFrame([model_name], columns=['model'])
    df.index = data.index
    data = pd.concat([df, data], axis=1)

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)
    
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(color='black')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    return ax, plt

#%%
seed_everything(1)

# -------------- base model parameters -----------------
TARGET = 'PanamaCity' # PanamaCity / London
WINDOW = 24 * 7
SAMPRATE = 1
STRIDE = 24
SCALE = 'STSC'
JSON_ROOT = '../DataInfo.JSON'
BATCH_SIZE = 128
MAX_EPOCHS = 200
IMAGE_PATH = '/home/mjy0750/Project/Result (scaler_batch_epoch)/' + SCALE + '_' + str(BATCH_SIZE) + '_' + str(MAX_EPOCHS) + '/'
ensure_dir(IMAGE_PATH)

with open(JSON_ROOT, 'rb') as f:
    DataInfo = json.load(f)
DIM = DataInfo[TARGET]['DIM']
# ------------------------------------------------------

# Load scaler
_ROOT_ = DataInfo['ROOT'] + TARGET
with open(_ROOT_ + '/x_scaler.pkl', 'rb') as f:
    x_scaler = pickle.load(f)[SCALE]
with open(_ROOT_ + '/y_scaler.pkl', 'rb') as f:
    y_scaler = pickle.load(f)[SCALE]

# Create custom dataset class from pre-processed data
dataset = TimeseriesPredictionDataset(
    TARGET_DATA=TARGET, 
    samprate=SAMPRATE, 
    window=WINDOW, 
    stride=STRIDE, 
    scaler=SCALE, 
    json_root=JSON_ROOT
)
# true value dataset (not scaled)
dataset_tv = TimeseriesPredictionDataset_TrueValue(
    TARGET_DATA=TARGET, 
    samprate=SAMPRATE, 
    window=WINDOW, 
    stride=STRIDE, 
    scaler=SCALE, 
    json_root=JSON_ROOT
)

# Split data into train(60%), val(20%), test(20%)
train_, test = train_test_split(dataset, test_size=0.2, shuffle=False)
train, val = train_test_split(train_, test_size=0.25, shuffle=False)

train_tv_, test_tv = train_test_split(dataset_tv, test_size=0.2, shuffle=False)
train_tv, val_tv = train_test_split(train_tv_, test_size=0.25, shuffle=False)

# Define loaders
train_loader = DataLoader(train, BATCH_SIZE)
val_loader = DataLoader(val, BATCH_SIZE)
test_loader = DataLoader(test, BATCH_SIZE)

#%%
lstm = fnn = mlp_mixer = None

#%%
# create model and train
lstm = create_model('LSTM')
#%%
fnn = create_model('FNN')
#%%
mlp_mixer = create_model('MLP_Mixer')

# %%
# Get Result of a Model

### Select Model
# 1. LSTM
# 2. FNN
# 3. MLP_Mixer
MODELS = {1:'LSTM', 2:'FNN', 3:'MLP_Mixer'}

# model = MODELS[1]
# visualize(model, start_index=0, steps=1) 
# visualize(model, start_index=len(test) - 1, steps=1)
# analysis(model)
# summarize(model)

for i in range(1, 4):
    model = MODELS[i]

    visualize(model, start_index=0, steps=1) 
    visualize(model, start_index=len(test) - 1, steps=1)

    analysis(model)
    summarize(model)

# %%
# Get Result of Whold Models
def visualize(start_index, steps):
    if start_index == 0: str_index = 'first'
    elif start_index + steps == len(test): str_index = 'last'
    else: str_index = str(start_index) + 'th'

    result_df = pd.DataFrame()
    for i in range(start_index, start_index + steps):
        lstm_df = pd.DataFrame(np.array(predict('LSTM', i)))
        fnn_df = pd.DataFrame(np.array(predict('FNN', i)))
        mlp_mixer_df = pd.DataFrame(np.array(predict('MLP_Mixer', i)))
        raw_df = test_tv[i][1]

        lstm_df.index = fnn_df.index = mlp_mixer_df.index = raw_df.index
        df = pd.concat([raw_df['load'], lstm_df, fnn_df, mlp_mixer_df], axis=1)
        result_df = pd.concat([result_df, df])
    
    result_df.columns = ['Raw Data', 'LSTM', 'FNN', 'MLP_Mixer']
    result_df.plot(figsize=(16, 5), legend=True, title='Prediction Graph for ' + str_index + ' index')

    plt.savefig(IMAGE_PATH + 'Predict_' + str_index + '_index.png')
    plt.show()

def analysis():
    models = ['LSTM', 'FNN', 'MLP_Mixer']

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), dpi=100)
    loss_df = pd.DataFrame()

    for i in range(len(models)):
        model_name = models[i]
        metrics = pd.read_csv('/home/mjy0750/Project/Log/' + model_name + '_Log/0/metrics.csv')
        train_loss = metrics[['train_loss', 'step', 'epoch']][~np.isnan(metrics['train_loss'])]
        val_loss = metrics[['val_loss', 'epoch']][~np.isnan(metrics['val_loss'])]
        test_loss = metrics['test_loss'].iloc[-1]

        axes[0].set_title('Train Loss')
        axes[0].plot(train_loss['epoch'], train_loss['train_loss'], label=model_name)
        axes[1].set_title('Validation Loss')
        axes[1].plot(val_loss['epoch'], val_loss['val_loss'], label=model_name)

        train_loss = train_loss['train_loss'].iloc[-1]
        val_loss = val_loss['val_loss'].iloc[-1]
        test_loss = test_loss
        loss = [train_loss, val_loss, test_loss]
        loss = np.reshape(loss, (1, 3))
        loss = pd.DataFrame(loss)

        loss_df = pd.concat([loss_df, loss])

    axes[0].legend()
    axes[1].legend()
    plt.savefig(IMAGE_PATH + 'Loss Graph.png')
    plt.show()
    loss_df.columns = ['Train Loss', 'Val Loss', 'Test Loss']
    loss_df.index = models
    
    _, loss_table = render_mpl_table(loss_df)
    loss_table.savefig(IMAGE_PATH + 'Loss Values.png')

def summarize():
    test_data = np.array(test)
    Y_test = test_data[:, 1]
    scores_lstm = []
    scores_fnn = []
    scores_mlp_mixer = []

    for i in range(len(Y_test)):
        Y_lstm, Y_fnn, Y_mlp_mixer = predict('LSTM', i), predict('FNN', i), predict('MLP_Mixer', i)

        Y = Y_test[i]
        Y = np.reshape(Y.detach().numpy(), (-1, 1))
        Y = y_scaler.inverse_transform(Y)
        Y = torch.Tensor(Y)

        scores_lstm.append(score(Y_lstm, Y))
        scores_fnn.append(score(Y_fnn, Y))
        scores_mlp_mixer.append(score(Y_mlp_mixer, Y))

    scores_lstm = np.array(scores_lstm).mean(axis=0)
    scores_fnn = np.array(scores_fnn).mean(axis=0)
    scores_mlp_mixer = np.array(scores_mlp_mixer).mean(axis=0)

    scores = np.concatenate([scores_lstm, scores_fnn, scores_mlp_mixer])
    scores = np.reshape(scores, [3, -1])

    df_scores = pd.DataFrame(scores)
    df_scores.columns = ['MAE', 'MAPE', 'MPE', 'MSE', 'RMSE', 'RRMSE']
    df_scores.index = ['LSTM', 'FNN', 'MLP_Mixer']
    
    _, score_table = render_mpl_table(df_scores)
    score_table.savefig(IMAGE_PATH + 'Score.png')

def render_mpl_table(data, col_width=2.0, row_height=0.625, font_size=14,
                     header_color='#A6A6A6', row_colors=['#f1f1f2', 'w'], edge_color='w',
                     bbox=[0, 0, 1, 1], header_columns=0,
                     ax=None, **kwargs):
    data.update(data[:].astype(float))
    data.update(data[:].applymap('{:,.3f}'.format))
    df = pd.DataFrame(['LSTM', 'FNN', 'MLP_Mixer'], columns=['model'])
    df.index = data.index
    data = pd.concat([df, data], axis=1)

    if ax is None:
        size = (np.array(data.shape[::-1]) + np.array([0, 1])) * np.array([col_width, row_height])
        fig, ax = plt.subplots(figsize=size)
        ax.axis('off')
    
    mpl_table = ax.table(cellText=data.values, bbox=bbox, colLabels=data.columns, cellLoc='center', **kwargs)
    
    mpl_table.auto_set_font_size(False)
    mpl_table.set_fontsize(font_size)

    for k, cell in  six.iteritems(mpl_table._cells):
        cell.set_edgecolor(edge_color)
        if k[0] == 0 or k[1] < header_columns:
            cell.set_text_props(color='black')
            cell.set_facecolor(header_color)
        else:
            cell.set_facecolor(row_colors[k[0]%len(row_colors) ])

    return ax, plt

# Function that returns the # of Parameters for each Model
def get_n_parameters():
    models = [lstm, fnn, mlp_mixer]
    nPrameters = []
    for model in models:
        pp = 0
        for p in list(model.parameters()):
            nn = 1
            for s in list(p.size()):
                nn = nn * s
            pp = pp + nn
        nPrameters.append(pp)

    return nPrameters # [lstm, fnn, mlp_mixer]

# Function that returns the Test Loss for each Model
def get_test_loss():
    models = ['LSTM', 'FNN', 'MLP_Mixer']
    loss = []
    for model in models:
        metrics = pd.read_csv('/home/mjy0750/Project/Log/' + model + '_Log/0/metrics.csv')
        test_loss = metrics['test_loss'].iloc[-1]
        loss.append(test_loss)

    return loss # [lstm, fnn, mlp_mixer]

# Function to visualize the test loss according to the # of parameters
def test_loss_per_nParameters():
    fig, ax = plt.subplots()
    x = get_n_parameters()
    y = get_test_loss()
    models = ['LSTM', 'FNN', 'MLP_Mixer']

    for i in range(len(x)):
        ax.plot(x[i], y[i], label=models[i], marker='o', linestyle='')

    ax.legend(fontsize=12)
    plt.title('Test Loss Per # of Parameters')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Loss')
    plt.savefig(IMAGE_PATH + 'Test Loss per # of Parameters.png')
    plt.show()

visualize(start_index=0, steps=3) 
visualize(start_index=len(test) - 3, steps=3)

analysis()
test_loss_per_nParameters()
summarize()

# %%
