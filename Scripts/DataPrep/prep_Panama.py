#%%
import glob, os
import pandas as pd
from pandas.core.tools.datetimes import Scalar 
pd.options.display.float_format = '{:,.2f}'.format 
import numpy as np
import matplotlib.pyplot as plt
import pickle
import tqdm
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

#%%
### Define root ###
TARGET_DATA = 'PanamaCity'
_ROOT_ = '/home/mjy0750/Project/Data/Processed/{}/'.format(TARGET_DATA)

### Data Loading ###
_BASE_ROOT_ = '/home/mjy0750/Project/Data/PanamaCity/RawData/'
df = pd.read_csv(_BASE_ROOT_ + 'dataset.csv')

### Object to Datetime & set index ###
if 'datetime' in df.columns:
    df['datetime'] = pd.to_datetime(df['datetime'])
df.set_index('datetime', inplace=True)

### Delete Unused Columns ###
df.drop(['T2M_san', 'QV2M_san', 'TQL_san', 'W2M_san', 'Holiday_ID',
            'T2M_dav', 'QV2M_dav', 'TQL_dav', 'W2M_dav'], axis=1, inplace=True)
### Rename Columns ###
df.columns = ['load', 'temperature', 'humidity', 'precipitation', 'wind_speed', 'holiday', 'school']

### Create Columns ###
# working_day column (1: working day, 0: weekend)
# Mon: 0, Tue: 1, Wed: 2, Thu: 3, Fri: 4, Sat: 5, Sun: 6 
working_day = np.array([0 if df.index[i].weekday() > 4 
                    else 1 
                    for i in range(len(df.index))]).T
df['working_day'] = working_day

# year, month, date, weekday column
df['year'] = df.index.year
df['month'] = df.index.month
df['day'] = df.index.day
df['weekday'] = df.index.weekday

df.asfreq('h')
# df.isnull().sum()

#%%
train_df_, test_df = train_test_split(df, test_size=0.2, shuffle=False)
train_df, val_df = train_test_split(train_df_, test_size=0.25, shuffle=False)

X_SCALERS = {'MMSC':MinMaxScaler(), 'STSC':StandardScaler()}
Y_SCALERS = {'MMSC':MinMaxScaler(), 'STSC':StandardScaler()}

X_SCALERS['MMSC'].fit(train_df)
X_SCALERS['STSC'].fit(train_df)
Y_SCALERS['MMSC'].fit(np.reshape(np.array(train_df['load']), (-1, 1)))
Y_SCALERS['STSC'].fit(np.reshape(np.array(train_df['load']), (-1, 1)))

# Save data to pickle
id = 0
PATH_TRAJECTORY = _ROOT_ + str(id) + '/'
ensure_dir(PATH_TRAJECTORY)
with open(PATH_TRAJECTORY + str(id) + '.pkl', 'wb') as f:
    pickle.dump(df, f)

with open(_ROOT_ + 'x_scaler.pkl', 'wb') as f:
    pickle.dump(X_SCALERS, f)
with open(_ROOT_ + 'y_scaler.pkl', 'wb') as f:
    pickle.dump(Y_SCALERS, f)
    
# %%
