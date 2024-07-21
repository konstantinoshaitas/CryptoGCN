import os
import numpy as np
import pandas as pd
from lstm_tf_class import CryptoLSTM
from pearson_correlation import CorrelationMatrix
from gcn_class import CryptoGCN

'''
1. LOAD DATA
'''

# Load using relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, '..', 'data')
sequential_data_dir = os.path.join(data_dir, 'processed_sequential_data')
results_dir = os.path.join(data_dir, 'tgcn_results')
os.makedirs(results_dir, exist_ok=True)


# List all CSV filenames in the sequential_data directory
assets = [f for f in os.listdir(sequential_data_dir) if f.endswith('.csv')]

'''
2. LSTM_EMBEDDING - GRAPH NODES
'''

hidden_states = []

# Iterate over the assets to get the hidden states
for asset in assets:
    asset_path = os.path.join(sequential_data_dir, asset)
    lstm_model = CryptoLSTM(csv_path=asset_path, num_epochs=5)
    lstm_model.run(manual_split=True)
    hidden_states_np = lstm_model.get_hidden_states(lstm_model.X)
    hidden_states.append(hidden_states_np)

# Convert list of hidden states to a numpy array
hidden_states = np.array(hidden_states)  # Shape: (num_assets, num_time_steps, hidden_state_dim)
print(hidden_states.shape)
hidden_states = np.transpose(hidden_states, (1, 0, 2))  # Shape: (num_time_steps, num_assets, hidden_state_dim)

print(f"Hidden states shape: {hidden_states.shape}")

'''
3. DENOISED PEARSON CORRELATION - GRAPH EDGES
'''
correlation_data_dir = os.path.join(data_dir, 'correlation_data', 'aggregated_asset_data.csv')
correlation_data = pd.read_csv(correlation_data_dir, index_col=0)

correlation_matrix = CorrelationMatrix(correlation_data, window_size=21)
denoised_matrices = correlation_matrix.run()

# Ensure denoised_matrices match the number of time steps in hidden_states
num_time_steps = hidden_states.shape[0]
denoised_matrices = denoised_matrices[:num_time_steps]

print(f"Number of denoised matrices: {len(denoised_matrices)}")
print(f"Shape of first denoised matrix: {denoised_matrices[0].shape}")

'''
4. TEMPORAL GRAPH MODEL
'''
gcn_model = CryptoGCN(denoised_matrices)
gcn_outputs = gcn_model.apply_gcn(hidden_states)

# Rank predictions for each time step
rankings = []
for output in gcn_outputs:
    ranking = np.argsort(output.flatten())[::-1]
    rankings.append(ranking)

'''
5. RESULTS ETC...
'''
output_dir = os.path.join(script_dir, '..', 'models')
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(results_dir, 'gcn_outputs.npy'), gcn_outputs)
np.save(os.path.join(results_dir, 'rankings.npy'), rankings)

print("GCN outputs and rankings have been saved in:", results_dir)
