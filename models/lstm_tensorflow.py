"""
PACKAGES
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
import random

'''
REPRODUCIBILITY
'''

random_seed = 42
np.random.seed(random_seed)
tf.random.set_seed(random_seed)
random.seed(random_seed)


'''
LOAD
'''

df = pd.read_csv(r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\INDEX_BTCUSD, 1D_43931.csv')
data = df
data.columns = data.columns.str.lower()
data = data.drop(columns="time")
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
DATA MANIPULATION
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


sequence_length = 14  # Number of prior data considered for next prediction



'''
NORMALISATION
'''

# scaler = MinMaxScaler()
# scaler = StandardScaler
scaler = MaxAbsScaler()
data_scaled = scaler.fit_transform(data)



'''
TRAIN TEST SPLIT
'''

X, y = sequences(data_scaled, sequence_length)
X_unscaled, y_unscaled = sequences(data, sequence_length)

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)
# NOTE: sklearn train test split does not work due to randomising sequential input-outputs

split_ratio = int(len(X) * 0.7)  # 70% train / test split

X_train, y_train, X_test, y_test = X[:split_ratio], y[:split_ratio], X[split_ratio:], y[split_ratio:]
X_unscaled_test, y_unscaled_test = X_unscaled[split_ratio:], y_unscaled[split_ratio:]

y_true = y_unscaled_test #actual prices from our test set

#TODO time series split sklearn for timeseries cross validation
'''
LSTM Model in Keras
'''
num_epochs = 16

model = Sequential()
model.add(LSTM(units=20, return_sequences=False, input_shape=(sequence_length, X.shape[2])))
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')  #COMPILE MODEL

model.fit(X_train, y_train, epochs=num_epochs, batch_size=32, validation_split=0.2, verbose=1)  #TRAIN MODEL

#TODO hyperparameters, layers, dense, hidden_nodes etc

'''
Testing & Plots
'''

predictions = model.predict(X_test)

#SCALE BACK THE PREDICTIONS
predicted_prices_scaled = np.zeros((len(predictions), data.shape[1])) # Create zero array for predicted prices
predicted_prices_scaled[:, 3] = predictions.flatten()  # Only the 'close' column (index 3)
predicted_prices = scaler.inverse_transform(predicted_prices_scaled)[:, 3]

true_prices_scaled = np.zeros((len(y_test), data.shape[1]))  # Create zero array for true prices
true_prices_scaled[:, 3] = y_test.flatten()  # Only the 'close' column (index 3)
true_prices = scaler.inverse_transform(true_prices_scaled)[:, 3]


#TODO show train and test error
#   test for overfitting


'''
PLOT
'''

plt.figure(figsize=(12, 8))
plt.xlabel('time', fontsize=13, fontweight='bold',labelpad=12)
plt.ylabel('prices', fontsize=13, fontweight='bold',labelpad=12)

plt.plot(predicted_prices, label='predicted prices', color=palette[4], linestyle='--')
plt.plot(true_prices, label='actual prices', color=palette[0])

plt.legend()
plt.show()
