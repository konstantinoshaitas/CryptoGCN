import os
import numpy as np
import pandas as pd
from lstm_tf_class import CryptoLSTM
from pearson_correlation import CorrelationMatrix
from gcn_class import CryptoGCN, apply_gcn
from gcn_visualizer import GCNVisualizer
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

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
train_y = []
test_y = []

for asset in assets:
    asset_path = os.path.join(sequential_data_dir, asset)
    print(f'Evaluating hidden states for: {asset_path}')
    lstm_model = CryptoLSTM(csv_path=asset_path, num_epochs=1)
    train_hs, test_hs = lstm_model.run()
    train_hidden_states.append(train_hs)
    test_hidden_states.append(test_hs)

    # Extract the target variable (e.g., next day's return) for training
    df = pd.read_csv(asset_path)
    returns = df['close'].pct_change().shift(-1).dropna()  # Next day's return
    train_y.append(returns[:len(train_hs)])
    test_y.append(returns[len(train_hs):len(train_hs) + len(test_hs)])

train_hidden_states = np.array(train_hidden_states)
test_hidden_states = np.array(test_hidden_states)
train_y = np.array(train_y)
test_y = np.array(test_y)

# Transpose the hidden states to match the expected shape
train_hidden_states = np.transpose(train_hidden_states, (1, 0, 2))
test_hidden_states = np.transpose(test_hidden_states, (1, 0, 2))
train_y = np.transpose(train_y)
test_y = np.transpose(test_y)

print("Shape of train_hidden_states:", train_hidden_states.shape)
print("Shape of test_hidden_states:", test_hidden_states.shape)
print("Shape of train_y:", train_y.shape)
print("Shape of test_y:", test_y.shape)

# Process correlation data
correlation_data_dir = os.path.join(data_dir, 'correlation_data', 'aggregated_asset_data.csv')
correlation_data = pd.read_csv(correlation_data_dir, index_col=0)

correlation_matrix = CorrelationMatrix(correlation_data, window_size=21)
train_denoised, test_denoised = correlation_matrix.run()

print("Shape of train_denoised:", np.array(train_denoised).shape)
print("Shape of test_denoised:", np.array(test_denoised).shape)

# Ensure all inputs have the same number of time steps
min_time_steps = min(train_hidden_states.shape[0], len(train_denoised), train_y.shape[0])
train_hidden_states = train_hidden_states[:min_time_steps]
train_denoised = np.array(train_denoised[:min_time_steps])
train_y = train_y[:min_time_steps]

min_time_steps = min(test_hidden_states.shape[0], len(test_denoised), test_y.shape[0])
test_hidden_states = test_hidden_states[:min_time_steps]
test_denoised = np.array(test_denoised[:min_time_steps])
test_y = test_y[:min_time_steps]

# Apply GCN
gcn_model, history, train_gcn_outputs, test_gcn_outputs = apply_gcn(
    train_hidden_states, test_hidden_states, train_denoised, test_denoised, train_y, test_y
)


# Generate rankings
def generate_rankings(outputs):
    return np.argsort(outputs, axis=1)[::-1]


train_rankings = generate_rankings(train_gcn_outputs)
test_rankings = generate_rankings(test_gcn_outputs)

# Save results
np.save(os.path.join(results_dir, 'train_gcn_outputs.npy'), train_gcn_outputs)
np.save(os.path.join(results_dir, 'test_gcn_outputs.npy'), test_gcn_outputs)
np.save(os.path.join(results_dir, 'train_rankings.npy'), train_rankings)
np.save(os.path.join(results_dir, 'test_rankings.npy'), test_rankings)

# Save the trained model
gcn_model.save(os.path.join(results_dir, 'trained_gcn_model.keras'))

print("GCN outputs, rankings, and training history have been saved in:", results_dir)

# Predict on test set
test_predictions = gcn_model.predict([test_hidden_states, test_denoised])

# Generate rankings for test predictions
test_rankings_pred = np.argsort(-test_predictions, axis=1)

# Generate true rankings from test_y
test_rankings_true = np.argsort(-test_y, axis=1)


def dcg_at_k(r, k):
    r = np.asfarray(r)[:k]
    if r.size:
        return np.sum(np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

def calculate_ndcg(y_true, y_pred, k=10):
    ndcg_scores = []
    for true, pred in zip(y_true, y_pred):
        r = [true[i] for i in np.argsort(pred)[::-1]]
        ndcg_scores.append(ndcg_at_k(r, k))
    return np.mean(ndcg_scores)

# Evaluate using Spearman's rank correlation
spearman_correlations = []
for pred_rank, true_rank in zip(test_rankings_pred, test_rankings_true):
    correlation, _ = spearmanr(pred_rank, true_rank)
    spearman_correlations.append(correlation)

average_correlation = np.mean(spearman_correlations)
print(f"Average Spearman's rank correlation: {average_correlation}")

# Calculate Mean Reciprocal Rank (MRR)
def mrr_score(y_true, y_pred):
    return np.mean([1. / (np.where(p == t[0])[0][0] + 1) for p, t in zip(y_pred, y_true)])

mrr = mrr_score(test_rankings_true[:, :1], test_rankings_pred)
print(f"Mean Reciprocal Rank: {mrr}")

# Calculate NDCG
ndcg_10 = calculate_ndcg(test_y, test_predictions, k=10)
print(f"NDCG@10: {ndcg_10}")

# Visualize rankings
plt.figure(figsize=(12, 6))
plt.plot(spearman_correlations)
plt.title("Spearman's Rank Correlation over Time")
plt.xlabel("Time Step")
plt.ylabel("Correlation")
plt.savefig(os.path.join(results_dir, 'spearman_correlation.png'))
plt.close()

# Save evaluation results
np.save(os.path.join(results_dir, 'test_rankings_pred.npy'), test_rankings_pred)
np.save(os.path.join(results_dir, 'test_rankings_true.npy'), test_rankings_true)

with open(os.path.join(results_dir, 'evaluation_results.txt'), 'w') as f:
    f.write(f"Average Spearman's rank correlation: {average_correlation}\n")
    f.write(f"Mean Reciprocal Rank: {mrr}\n")
    f.write(f"NDCG@10: {ndcg_10}\n")

print("Evaluation results have been saved.")
