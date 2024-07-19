import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MaxAbsScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt
import random


class CryptoLSTM:
    def __init__(self, csv_path, sequence_length=14, num_epochs=1, random_seed=42,
                 date_col='time', target_col='close'):
        self.csv_path = csv_path
        self.sequence_length = sequence_length
        self.num_epochs = num_epochs
        self.random_seed = random_seed
        self.date_col = date_col
        self.target_col = target_col
        self.model = None
        self.trained_model = None
        self.scaler = MaxAbsScaler()
        self.set_random_seed()

    def set_random_seed(self):
        np.random.seed(self.random_seed)
        tf.random.set_seed(self.random_seed)
        random.seed(self.random_seed)

    def load_data(self):
        df = pd.read_csv(self.csv_path)
        df.columns = df.columns.str.lower()
        df[self.date_col] = pd.to_datetime(df[self.date_col])
        self.dates = df[self.date_col].values
        self.date_sequences = [self.dates[i + self.sequence_length] for i in
                               range(len(self.dates) - self.sequence_length)]
        self.data = df.drop(columns=self.date_col)
        # print(self.data.info())
        # print(self.data.head())

    def add_technical_indicators(self):
        def rma(pandaseries: pd.Series, period: int) -> pd.Series:
            return pandaseries.ewm(alpha=1 / period).mean()

        def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
            high, low, prev_close = df['high'], df['low'], df['close'].shift()
            true_range_all = [high - low, high - prev_close, low - prev_close]
            true_range_all = [tr.abs() for tr in true_range_all]
            tr = pd.concat(true_range_all, axis=1).max(axis=1)
            return rma(tr, length)

        def rsi(df: pd.DataFrame, length: int = 14) -> pd.Series:
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = rma(gain, length)
            avg_loss = rma(loss, length)
            rs = avg_gain / avg_loss
            return 100 - (100 / (1 + rs))

        self.data["RSI"] = rsi(self.data)
        self.data["ATR"] = atr(self.data)
        self.drop_null_columns()
        self.data = self.data.dropna()

    def drop_null_columns(self, threshold=0.8):
        for column in self.data.columns:
            if self.data[column].isna().mean() >= threshold:
                self.data.drop(columns=[column], inplace=True)

    def preprocess_data(self):
        self.data_scaled = self.scaler.fit_transform(self.data)
        self.X, self.y = self.create_sequences(self.data_scaled, self.sequence_length,
                                               self.data.columns.get_loc(self.target_col))
        self.X_unscaled, self.y_unscaled = self.create_sequences(self.data.values, self.sequence_length,
                                                                 self.data.columns.get_loc(self.target_col))

    def create_sequences(self, data, sequence_length, target_col_index):
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            x = data[i:(i + sequence_length)]
            y = data[i + sequence_length][target_col_index]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def train_test_split(self):
        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(self.X):
            self.X_train, self.X_test = self.X[train_index], self.X[test_index]
            self.y_train, self.y_test = self.y[train_index], self.y[test_index]
            self.X_unscaled_train, self.X_unscaled_test = self.X_unscaled[train_index], self.X_unscaled[test_index]
            self.y_unscaled_train, self.y_unscaled_test = self.y_unscaled[train_index], self.y_unscaled[test_index]
            self.y_true = self.y_unscaled_test
            self.test_dates = [self.date_sequences[i] for i in test_index]

    def manual_train_test_split(self, split_ratio=0.75):
        split_index = int(len(self.X) * split_ratio)
        self.X_train, self.X_test = self.X[:split_index], self.X[split_index:]
        self.y_train, self.y_test = self.y[:split_index], self.y[split_index:]
        self.X_unscaled_train, self.X_unscaled_test = self.X_unscaled[:split_index], self.X_unscaled[split_index:]
        self.y_unscaled_train, self.y_unscaled_test = self.y_unscaled[:split_index], self.y_unscaled[split_index:]
        self.y_true = self.y_unscaled_test
        self.test_dates = self.dates[self.sequence_length + split_index:self.sequence_length + len(self.X)]

    def build_model(self):
        inputs = Input(shape=(self.sequence_length, self.X.shape[2]))
        lstm_out = LSTM(units=40, return_sequences=False, return_state=True, kernel_regularizer=l2(0.01))
        lstm_out, state_h, state_c = lstm_out(inputs)
        dropout_1 = Dropout(0.2)(lstm_out)
        dense_1 = Dense(units=20)(dropout_1)
        dropout_2 = Dropout(0.2)(dense_1)
        outputs = Dense(units=1)(dropout_2)
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.lstm_hidden_layer = Model(inputs=inputs, outputs=state_h)  # Model to get hidden states

    def train_model(self):
        self.trained_model = self.model.fit(self.X_train, self.y_train, epochs=self.num_epochs, batch_size=32,
                                            validation_split=0.15, verbose=1)

    def evaluate_model(self):
        # Training performance
        train_pred = self.model.predict(self.X_train)

        # Create a dummy array with the same shape as the original data
        dummy_train = np.zeros((len(train_pred), self.data.shape[1]))
        dummy_train[:, self.data.columns.get_loc(self.target_col)] = train_pred.flatten()

        # Inverse transform
        train_pred_unscaled = self.scaler.inverse_transform(dummy_train)[:, self.data.columns.get_loc(self.target_col)]

        train_mse = np.mean((train_pred_unscaled - self.y_unscaled_train) ** 2)
        train_rmse = np.sqrt(train_mse)
        train_mae = np.mean(np.abs(train_pred_unscaled - self.y_unscaled_train))

        # Test performance
        test_pred = self.model.predict(self.X_test)

        # Create a dummy array with the same shape as the original data
        dummy_test = np.zeros((len(test_pred), self.data.shape[1]))
        dummy_test[:, self.data.columns.get_loc(self.target_col)] = test_pred.flatten()

        # Inverse transform
        test_pred_unscaled = self.scaler.inverse_transform(dummy_test)[:, self.data.columns.get_loc(self.target_col)]

        test_mse = np.mean((test_pred_unscaled - self.y_unscaled_test) ** 2)  # Mean Squared Error
        test_rmse = np.sqrt(test_mse)  # Root Mean Squared Error
        test_mae = np.mean(np.abs(test_pred_unscaled - self.y_unscaled_test))  # Mean Absolute Error

        print(f'Training - MSE: {train_mse:.2f}, RMSE: {train_rmse:.2f}, MAE: {train_mae:.2f}')
        print(f'Test - MSE: {test_mse:.2f}, RMSE: {test_rmse:.2f}, MAE: {test_mae:.2f}')

        # Store these metrics if you want to use them later
        self.train_metrics = {'MSE': train_mse, 'RMSE': train_rmse, 'MAE': train_mae}
        self.test_metrics = {'MSE': test_mse, 'RMSE': test_rmse, 'MAE': test_mae}

    def make_predictions(self):
        self.predictions = self.model.predict(self.X_test)
        predicted_prices_scaled = np.zeros((len(self.predictions), self.data.shape[1]))
        predicted_prices_scaled[:, self.data.columns.get_loc(self.target_col)] = self.predictions.flatten()
        true_prices_scaled = np.zeros((len(self.y_test), self.data.shape[1]))
        true_prices_scaled[:, self.data.columns.get_loc(self.target_col)] = self.y_test.flatten()
        self.predicted_prices = self.scaler.inverse_transform(predicted_prices_scaled)[:,
                                self.data.columns.get_loc(self.target_col)]
        self.true_prices = self.scaler.inverse_transform(true_prices_scaled)[:,
                           self.data.columns.get_loc(self.target_col)]

    def display(self):
        sns.set_style("white")
        sns.set_context("talk", font_scale=.8)
        palette = sns.color_palette()

        plt.figure(figsize=(12, 8))
        plt.plot(self.trained_model.history['loss'], label='Training Loss')
        plt.plot(self.trained_model.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        plt.show()

        plt.figure(figsize=(12, 8))
        plt.xlabel('Dates', fontsize=13, fontweight='bold', labelpad=12)
        plt.ylabel('Prices', fontsize=13, fontweight='bold', labelpad=12)
        plt.plot(self.test_dates, self.predicted_prices, label='Predicted Prices', color=palette[4], linestyle='--')
        plt.plot(self.test_dates, self.y_true, label='Actual Prices', color=palette[0])
        plt.title('Predicted vs True Prices')
        plt.legend()
        plt.show()

    def get_hidden_states(self, data):
        hidden_state = self.lstm_hidden_layer.predict(data)
        return hidden_state

    def visualize_hidden_states(self, data):
        hidden_states = self.get_hidden_states(data)

        # Apply PCA to reduce dimensions to 2 for visualization
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(hidden_states)

        # Alternatively, you can use t-SNE for better separation in some cases
        tsne = TSNE(n_components=2)
        tsne_result = tsne.fit_transform(hidden_states)

        plt.figure(figsize=(12, 8))
        plt.scatter(pca_result[:, 0], pca_result[:, 1], c=self.y, cmap='viridis')
        plt.colorbar()
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title('PCA of LSTM Hidden States')
        plt.show()

        # To use t-SNE, uncomment the following lines:
        plt.figure(figsize=(12, 8))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=self.y, cmap='viridis')
        plt.colorbar()
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.title('t-SNE of LSTM Hidden States')
        plt.show()

    def visualize_hidden_states_heatmap(self, data):
        hidden_states = self.get_hidden_states(data)
        plt.figure(figsize=(12, 8))
        sns.heatmap(hidden_states, cmap='viridis')
        plt.xlabel('Hidden Units')
        plt.ylabel('Time Steps')
        plt.title('Heatmap of LSTM Hidden States')
        plt.show()

    def visualize_hidden_states_lineplot(self, data):
        hidden_states = self.get_hidden_states(data)
        plt.figure(figsize=(12, 8))

        # Plot only every 5th hidden unit to reduce clutter
        for i in range(0, hidden_states.shape[1], 5):
            plt.plot(hidden_states[:, i], label=f'Hidden Unit {i + 1}')

        plt.xlabel('Time Steps')
        plt.ylabel('Hidden State Value')
        plt.title('Line Plot of LSTM Hidden States')
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        plt.show()

    def run(self, manual_split=False):
        self.load_data()
        self.add_technical_indicators()
        self.preprocess_data()
        if manual_split:
            self.manual_train_test_split()
        else:
            self.train_test_split()
        self.build_model()
        self.train_model()
        self.evaluate_model()
        self.make_predictions()
        # self.display()


# Test usage
if __name__ == "__main__":
    crypto_model = CryptoLSTM(csv_path=r'C:\Users\Kosta\Desktop\THESIS\CryptoGCN\data\INDEX_BTCUSD, 1D_43931.csv')
    crypto_model.run(manual_split=True)
    hidden_states = crypto_model.get_hidden_states(crypto_model.X)
    crypto_model.visualize_hidden_states(crypto_model.X)
    crypto_model.visualize_hidden_states_heatmap(crypto_model.X)
    crypto_model.visualize_hidden_states_lineplot(crypto_model.X)
    print(hidden_states.shape)
