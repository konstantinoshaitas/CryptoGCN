import os
import pandas as pd
from lstm_tf_class import CryptoLSTM
from pearson_correlation import CorrelationMatrix
from gcn_class import CryptoGCN

# Define paths using relative paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
sequential_data_dir = os.path.join(data_dir, 'sequential_data')

# List all CSV filenames in the sequential_data directory
assets = [f for f in os.listdir(sequential_data_dir) if f.endswith('.csv')]

hidden_states = []

# Iterate over the assets to get the hidden states
for asset in assets:
    asset_path = os.path.join(sequential_data_dir, asset)
    lstm_model = CryptoLSTM(csv_path=asset_path)
    lstm_model.run(manual_split=True)
    hidden_states_np = lstm_model.get_hidden_states(lstm_model.X)

    # Convert numpy array to DataFrame and add to the list
    hidden_states_df = pd.DataFrame(hidden_states_np)
    hidden_states.append(hidden_states_df)

# Concatenate hidden states DataFrames
hidden_states_df = pd.concat(hidden_states, axis=1)

# Load the aggregated asset data CSV
correlation_data_path = os.path.join(data_dir, 'correlation_data', 'aggregated_asset_data.csv')
correlation_data = pd.read_csv(correlation_data_path, index_col=0)

# Initialize the CorrelationMatrix class
correlation_matrix = CorrelationMatrix(correlation_data, window_size=21)

# Calculate returns
correlation_matrix.calculate_returns()

# Calculate volatility
correlation_matrix.calculate_volatility()

# Compute rolling correlations
correlation_matrix.compute_rolling_correlations()

# Compute eigenvalues and eigenvectors
correlation_matrix.compute_eigenvalues_eigenvectors()

# Denoise the correlation matrices
correlation_matrix.denoise_correlation_matrices()

denoised_matrices = correlation_matrix.denoised_matrices

# Initialize the GCN class with denoised correlation matrices
gcn_model = CryptoGCN(denoised_matrices)

# Apply the GCN to the hidden states
gcn_outputs = gcn_model.apply_gcn(hidden_states_df.values)

# Example: Perform predictions with the trained GCN model
print(gcn_outputs)

# Optionally: Save results or models
output_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(output_dir, exist_ok=True)
for i, output in enumerate(gcn_outputs):
    pd.DataFrame(output).to_csv(os.path.join(output_dir, f'gcn_output_{i}.csv'))
