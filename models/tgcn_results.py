import numpy as np
import matplotlib.pyplot as plt
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'data')
results_dir = os.path.join(data_dir, 'tgcn_results')

train_gcn_outputs = np.load(os.path.join(results_dir, 'train_gcn_outputs.npy'))
test_gcn_outputs = np.load(os.path.join(results_dir, 'test_gcn_outputs.npy'))
train_rankings = np.load(os.path.join(results_dir, 'train_rankings.npy'))
test_rankings = np.load(os.path.join(results_dir, 'test_rankings.npy'))

print("Train GCN outputs shape:", train_gcn_outputs.shape)
print("Test GCN outputs shape:", test_gcn_outputs.shape)
print("Train Rankings shape:", train_rankings.shape)
print("Test Rankings shape:", test_rankings.shape)

print("Train Rankings sample:", train_rankings[0])
print("Test Rankings sample:", test_rankings[0])

print("Train GCN outputs min and max:", train_gcn_outputs.min(), train_gcn_outputs.max())
print("Test GCN outputs min and max:", test_gcn_outputs.min(), test_gcn_outputs.max())
print("Train Rankings min and max:", train_rankings.min(), train_rankings.max())
print("Test Rankings min and max:", test_rankings.min(), test_rankings.max())
print("Unique train ranking values:", np.unique(train_rankings))
print("Unique test ranking values:", np.unique(test_rankings))

def plot_rankings(rankings, title):
    crypto_index = 7  # Change this to look at different cryptocurrencies
    plt.figure(figsize=(12, 6))
    plt.plot(rankings[:, crypto_index])
    plt.title(f"{title} - Rankings over time for cryptocurrency {crypto_index}")
    plt.xlabel("Time step")
    plt.ylabel("Rank")
    plt.show()

plot_rankings(train_rankings, "Train")
plot_rankings(test_rankings, "Test")

def plot_rank_distributions_over_time(rankings, save_dir, segment_size=100, title_prefix=""):
    num_segments = len(rankings) // segment_size
    num_cryptos = rankings.shape[1]

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        segment_rankings = rankings[start:end]

        plt.figure(figsize=(15, 10))
        plt.boxplot([segment_rankings[:, j] for j in range(num_cryptos)])
        plt.title(f"{title_prefix} Distribution of Rankings (Time steps {start} to {end})")
        plt.xlabel("Cryptocurrency")
        plt.ylabel("Rank")
        plt.ylim(0, num_cryptos)
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_rank_distribution_segment_{i + 1}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"Created {num_segments} visualizations in {save_dir}")

def plot_rankings_line_over_time(rankings, save_dir, segment_size=500, title_prefix=""):
    num_segments = len(rankings) // segment_size
    num_cryptos = rankings.shape[1]

    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        segment_rankings = rankings[start:end]

        plt.figure(figsize=(20, 10))
        for j in range(num_cryptos):
            plt.plot(range(start, end), segment_rankings[:, j], label=f'Crypto {j}')

        plt.title(f"{title_prefix} Rankings for Each Cryptocurrency (Time steps {start} to {end})")
        plt.xlabel("Time step")
        plt.ylabel("Rank")
        plt.ylim(0, num_cryptos)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        save_path = os.path.join(save_dir, f"{title_prefix.lower().replace(' ', '_')}_rankings_line_segment_{i + 1}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"Created {num_segments} line plot visualizations in {save_dir}")

# Line plot of ranking over time
save_directory = os.path.join(data_dir, results_dir, 'ranking_line_plots')
plot_rankings_line_over_time(train_rankings, save_directory, segment_size=20, title_prefix="Train")
plot_rankings_line_over_time(test_rankings, save_directory, segment_size=20, title_prefix="Test")

# Rank distribution box plot over time
save_directory = os.path.join(data_dir, results_dir, 'ranking_distribution')
plot_rank_distributions_over_time(train_rankings, save_directory, segment_size=20, title_prefix="Train")
plot_rank_distributions_over_time(test_rankings, save_directory, segment_size=20, title_prefix="Test")

# Further visualizations or analyses here as needed