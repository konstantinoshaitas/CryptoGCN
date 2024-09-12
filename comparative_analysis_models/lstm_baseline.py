import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def build_lstm_baseline(hp, sequence_length, num_assets):
    model = Sequential()
    lstm_units = hp.Int('lstm_units', min_value=10, max_value=100, step=10)
    model.add(LSTM(units=lstm_units, input_shape=(sequence_length, num_assets)))
    model.add(Dense(units=num_assets, activation='linear'))
    learning_rate = hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='log')
    opt = Adam(learning_rate=learning_rate, clipvalue=1.0)
    model.compile(optimizer=opt, loss='mean_squared_error')

    return model
