import os
import numpy as np
import pandas as pd
from lstm_tf_class import CryptoLSTM
from pearson_correlation import CorrelationMatrix
from gcn_class import CryptoGCN
from gcn_visualizer import GCNVisualizer

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

# Process LSTM data
train_hidden_states = []
test_hidden_states = []

for asset in assets:
    asset_path = os.path.join(sequential_data_dir, asset)
    print(f'Evaluating hidden states for: {asset_path}')
    lstm_model = CryptoLSTM(csv_path=asset_path)
    train_hs, test_hs = lstm_model.run()
    train_hidden_states.append(train_hs)
    test_hidden_states.append(test_hs)

train_hidden_states = np.array(train_hidden_states)
test_hidden_states = np.array(test_hidden_states)

# Transpose the hidden states to match the expected shape
train_hidden_states = np.transpose(train_hidden_states, (1, 0, 2))
test_hidden_states = np.transpose(test_hidden_states, (1, 0, 2))

# Process correlation data
correlation_data_dir = os.path.join(data_dir, 'correlation_data', 'aggregated_asset_data.csv')
correlation_data = pd.read_csv(correlation_data_dir, index_col=0)

correlation_matrix = CorrelationMatrix(correlation_data, window_size=21)
train_denoised, test_denoised = correlation_matrix.run()
# correlation_matrix.plot_correlation_matrices(start=3000, end=3002)
# Visualize a single graph
GCNVisualizer.visualize_graph(test_denoised[0], save_path=os.path.join(results_dir, 'gcn_graph.png'))

# Visualize graph over time
# GCNVisualizer.visualize_graph_over_time(train_denoised, interval=100,
#                                         save_path=os.path.join(results_dir, 'gcn_graph_time'))

# Apply GCN
gcn_model = CryptoGCN(train_denoised, test_denoised)
train_gcn_outputs, test_gcn_outputs = gcn_model.apply_gcn(train_hidden_states, test_hidden_states, batch_size=16)


# Generate rankings
def generate_rankings(outputs):
    return np.argsort(outputs.squeeze(), axis=1)[::-1]


train_rankings = generate_rankings(train_gcn_outputs)
test_rankings = generate_rankings(test_gcn_outputs)

# Save results
np.save(os.path.join(results_dir, 'train_gcn_outputs.npy'), train_gcn_outputs)
np.save(os.path.join(results_dir, 'test_gcn_outputs.npy'), test_gcn_outputs)
np.save(os.path.join(results_dir, 'train_rankings.npy'), train_rankings)
np.save(os.path.join(results_dir, 'test_rankings.npy'), test_rankings)

print("GCN outputs and rankings have been saved in:", results_dir)
