import os
import pandas as pd
import numpy as np
from lstm_tf_class import CryptoLSTM
from pearson_correlation import CorrelationMatrix
from gcn_class import CryptoGCN

def run_tgcn(crypto_csv_paths, asset_data_csv_path):
    hidden_states_list = []

    # 1: Process each crypto asset CSV file
    for crypto_csv_path in crypto_csv_paths:
        print(f"Processing {crypto_csv_path}")
        crypto_lstm = CryptoLSTM(csv_path=crypto_csv_path)
        crypto_lstm.run(manual_split=True)
        hidden_states = crypto_lstm.get_hidden_states(crypto_lstm.X)
        hidden_states_list.append(hidden_states)

    # Combine hidden states from all assets
    combined_hidden_states = np.concatenate(hidden_states_list, axis=0)

    # 2: Load multiple assets data for correlation matrix calculation
    asset_data = pd.read_csv(asset_data_csv_path)
    correlation_matrix = CorrelationMatrix(asset_data)

    # Calculate returns and volatility
    correlation_matrix.calculate_returns()
    correlation_matrix.calculate_volatility()

    # 3: Calculate denoised correlation matrices
    denoised_matrices = correlation_matrix.calculate_denoised_correlation_matrices(method='returns')

    # 4: Apply GCN to hidden states using the denoised correlation matrices  #TODO: WORK FROM HERE
    crypto_gcn = CryptoGCN(denoised_matrices)
    gcn_outputs = crypto_gcn.apply_gcn(combined_hidden_states)

    # Optionally: Implement backtesting or further analysis on gcn_outputs
    print("GCN Outputs:\n", gcn_outputs)

# test usage
crypto_csv_paths = [
    r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\INDEX_BTCUSD, 1D_43931.csv',
    # Add paths to other crypto asset CSV files here
]

# Data with daily/8hr/4hr returns of all assets. Each column should represent an asset.
asset_data_csv_path = r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\asset_data.csv'

run_tgcn(crypto_csv_paths, asset_data_csv_path)
