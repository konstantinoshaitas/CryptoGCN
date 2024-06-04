"""
PACKAGES
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import random

'''
REPRODUCIBILITY AND PARAMETERS
'''

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)
sequence_length = 21  # Number of prior data considered for next prediction

'''
LOAD
'''

df = pd.read_csv(r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\INDEX_BTCUSD, 1D_43931.csv')
data = df
data.columns = data.columns.str.lower()


data['time'] = pd.to_datetime(data['time'])
dates = data['time'].values
date_sequences = [dates[i + sequence_length] for i in range(len(dates) - sequence_length)] #create data sequences for plots

data = data.drop(columns="time") #drop dates from dataset for model train, pred, and eval
print(data.info())
print(data.head())
data = data.values
# print(len(data))
# print(data.shape)
# print(data[0])

'''
AESTHETIC
'''

sns.set_style("white")
sns.set_context("talk", font_scale=.8)
palette = sns.color_palette()


'''
CREATE SEQUENTIAL DATA
'''

def sequences(data, sequence_length):
    xs, ys = [], []
    loop = len(data) - sequence_length
    for i in range(loop):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length][3]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)


'''
NORMALISERS
'''

# scaler = MinMaxScaler()
# scaler = StandardScaler
scaler = MaxAbsScaler()

data_scaled = scaler.fit_transform(data)


'''
GENERATE X, y SEQUENCES
'''

X, y = sequences(data_scaled, sequence_length)
X_unscaled, y_unscaled = sequences(data, sequence_length)


'''
TRAIN TEST SPLIT
'''

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)
# NOTE1: sklearn train test split does not work due to randomising sequential input-outputs

tscv = TimeSeriesSplit(n_splits=5) #NOTE2: TIMESERIESSPLIT FROM SKLEARN FOR CROSS-VAL
for train_index, test_index in tscv.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    X_unscaled_train, X_unscaled_test = X_unscaled[train_index], X_unscaled[test_index]
    y_unscaled_train, y_unscaled_test = y_unscaled[train_index], y_unscaled[test_index]
    y_true = y_unscaled_test  # actual prices from our test set
    test_dates = [date_sequences[i] for i in test_index]


# split_ratio = int(len(X) * 0.7)  # 70% train / test split -- #NOTE3: MANUAL SPLIT
#
# X_train, y_train, X_test, y_test = X[:split_ratio], y[:split_ratio], X[split_ratio:], y[split_ratio:]
# X_unscaled_test, y_unscaled_test = X_unscaled[split_ratio:], y_unscaled[split_ratio:]
# y_true = y_unscaled_test #actual prices from our test set




'''
LSTM MODEL KERAS
'''
num_epochs = 32

model = Sequential()
model.add(LSTM(units=40, return_sequences=False, input_shape=(sequence_length, X.shape[2]), kernel_regularizer=l2(0.01)))
model.add(Dropout(0.2))
model.add(Dense(units=20))
model.add(Dropout(0.3))
model.add(Dense(units=1))


model.compile(optimizer='adam', loss='mean_squared_error')  #COMPILE MODEL

trained_model = model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, validation_split=0.1, verbose=1)  #TRAIN MODEL

test_loss = model.evaluate(X_test, y_test, verbose=0)
train_loss = model.evaluate(X_train, y_train, verbose=0)
print(f'Test Loss: {test_loss}')
print(f'Train Loss: {train_loss}')

#TODO hyperparameters, layers, dense, hidden_nodes etc

'''
PREDICTIONS
'''

predictions = model.predict(X_test)

#SCALE BACK THE PREDICTIONS
predicted_prices_scaled = np.zeros((len(predictions), data.shape[1])) # Create zero array for predicted prices
predicted_prices_scaled[:, 3] = predictions.flatten()  # Only the 'close' column (index 3)
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)[:, 3]

true_prices_scaled = np.zeros((len(y_test), data.shape[1]))  # Create zero array for true prices
true_prices_scaled[:, 3] = y_test.flatten()  # Only the 'close' column (index 3)
true_prices = scaler.inverse_transform(true_prices_scaled)[:, 3]


'''
PLOTS
'''
#Training and validation Loss
plt.figure(figsize=(12, 8))
plt.plot(trained_model.history['loss'], label='Training Loss')
plt.plot(trained_model.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

#Predictions on Test Data
plt.figure(figsize=(12, 8))
plt.xlabel('Dates', fontsize=13, fontweight='bold',labelpad=12)
plt.ylabel('Prices', fontsize=13, fontweight='bold',labelpad=12)
plt.plot(test_dates, predicted_prices, label='predicted prices', color=palette[4], linestyle='--')
plt.plot(test_dates, y_true, label='actual prices', color=palette[0])
plt.title('Predicted vs True Prices')
plt.legend()
plt.show()
