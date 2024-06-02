#PACKAGES

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import matplotlib.pyplot as plt


# LOAD

df = pd.read_csv(r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\INDEX_BTCUSD, 1D_43931.csv')
closes = df['close'].values.reshape(-1, 1)




# NORMALISATION

# scaler = MinMaxScaler()
# scaler = StandardScaler
scaler = MaxAbsScaler()
closes_scaled = scaler.fit_transform(closes)




# DATA MANIPULATION

def sequences(data, sequence_length):
    xs, ys = [], []
    loop = len(data) - sequence_length
    for i in range(loop):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 100  # Number of prior data considered for next prediction
X, y = sequences(closes_scaled, sequence_length)



#TRAIN TEST SPLIT

# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, test_size=0.3)
# NOTE: sklearn train test split does not work due to randomising sequential input-outputs

split_ratio = int(len(closes_scaled) * 0.7) ## 70% train / test split

X_train, y_train = X[:split_ratio], y[:split_ratio]
X_test, y_test = X[split_ratio:], y[split_ratio:]



class LSTM_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM_Model, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_() # init hidden state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_() # init cell state
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))                       #forward propagate
        out = out[:, -1, :]
        out = self.linear(out)
        return out



#MODEL PARAMETERS

input_dim = 1
hidden_dim = 20
num_layers = 1
output_dim = 1

model = LSTM_Model(input_dim, hidden_dim, num_layers, output_dim)



#CREATE TENSORS FROM X and y

X_train_tensors = torch.Tensor(X_train).float()
y_train_tensors = torch.tensor(y_train).float()
X_test_tensors = torch.tensor(X_test).float()
y_test_tensors = torch.tensor(y_test).float()


#LOSS

measure_loss = nn.MSELoss()
optim = torch.optim.Adam(model.parameters(), lr=0.03)



#TRAINING

num_epochs = 100

for epochs in range(num_epochs):
    model.train()
    optim.zero_grad()
    y_pred = model(X_train_tensors)
    MSE = measure_loss(y_pred, y_train_tensors)
    MSE.backward()
    optim.step()

    if epochs % 10 == 0:
        print(f' Epoch{epochs}, Loss:{MSE.item()}')



#TESTING & PLOTS

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensors)
    predictions_np = predictions.numpy()
    predictions_np = predictions_np.reshape(-1, 1)
    predicted_prices = scaler.inverse_transform(predictions_np)

true_prices = scaler.inverse_transform(y_test)

plt.figure(figsize=(4, 4))
plt.plot(predicted_prices, label='predictions', color='red', linestyle='--')
plt.plot(true_prices, label='true', color='blue')

plt.xlabel('time')
plt.ylabel('prices')
plt.legend()
plt.show()
