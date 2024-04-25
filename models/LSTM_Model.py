import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# Load data
df = pd.read_csv(r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\INDEX_BTCUSD, 1D_43931.csv')       # Update path as necessary
closes = df['close'].values.reshape(-1, 1)



# Normalize data
scaler = StandardScaler()
closes_scaled = scaler.fit_transform(closes)


# Create data sequences for LSTM training
def sequences(data, sequence_length):
    xs, ys = [], []
    loop = len(data) - sequence_length
    for i in range(loop):
        x = data[i:(i + sequence_length)]
        y = data[i + sequence_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

sequence_length = 10  # Number of Data used for next prediction
X, y = sequences(closes_scaled, sequence_length)


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
hidden_dim = 50
num_layers = 1
output_dim = 1

model = LSTM_Model(input_dim, hidden_dim, num_layers, output_dim)


#CREATE TENSORS FROM X and y

X_train_tensors = torch.Tensor(X).float()
y_train_tensors = torch.tensor(y).float()


#LOSS

measure_loss = nn.MSELoss()
opt = torch.optim.Adam(model.parameters(), lr=0.03)


#TRAINING LOOP

num_epochs = 50

for epochs in range(num_epochs):
    model.train()
    opt.zero_grad()
    y_pred = model(X_train_tensors)
    MSE = measure_loss(y_pred, y_train_tensors)
    MSE.backward()
    opt.step()

    if epochs % 10 == 0:
        print(f' Epoch{epochs}, Loss:{MSE.item()}')



