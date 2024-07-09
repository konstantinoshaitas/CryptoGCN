import pandas as pd
import numpy as np
from lstm_tf_class import CryptoLSTM
from pearson_correlation import CorrelationMatrix
from gcn_class import CryptoGCN

def run_tgcn(crypto_csv_path, asset_data_csv_path):
    # Step 1: Load and preprocess data using CryptoLSTM
    crypto_lstm = CryptoLSTM(csv_path=crypto_csv_path)
    crypto_lstm.run()

    # Step 2: Get hidden states from LSTM
    hidden_states = crypto_lstm.get_hidden_states(crypto_lstm.X)

    # Step 3: Load multiple assets data for correlation matrix calculation
    asset_data = pd.read_csv(asset_data_csv_path)
    correlation_matrix = CorrelationMatrix(asset_data)

    # Calculate returns and volatilities
    correlation_matrix.calculate_returns()
    correlation_matrix.calculate_volatility()

    # Step 4: Calculate denoised correlation matrices
    denoised_matrices = correlation_matrix.calculate_denoised_correlation_matrices(method='returns')

    # Step 5: Apply GCN to hidden states using the denoised correlation matrices
    crypto_gcn = CryptoGCN(denoised_matrices)
    gcn_outputs = crypto_gcn.apply_gcn(hidden_states)

    # Optionally: Implement backtesting or further analysis on gcn_outputs
    print("GCN Outputs:\n", gcn_outputs)

# Example usage
crypto_csv_path = r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\INDEX_BTCUSD, 1D_43931.csv'
asset_data_csv_path = r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\asset_data.csv'
run_tgcn(crypto_csv_path, asset_data_csv_path)
