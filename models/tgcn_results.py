import numpy as np
import matplotlib.pyplot as plt
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
data_dir = os.path.join(project_root, 'data')
results_dir = os.path.join(data_dir, 'tgcn_results')

gcn_outputs = np.load(os.path.join(results_dir, 'gcn_outputs.npy'))
rankings = np.load(os.path.join(results_dir, 'rankings.npy'))

print("GCN outputs shape:", gcn_outputs.shape)
print("Rankings shape:", rankings.shape)

print(rankings[3000])

print("GCN outputs min and max:", gcn_outputs.min(), gcn_outputs.max())
print("Rankings min and max:", rankings.min(), rankings.max())
print("Unique ranking values:", np.unique(rankings))

crypto_index = 7  # Change this to look at different cryptocurrencies
plt.plot(rankings[:, crypto_index])
plt.title(f"Rankings over time for cryptocurrency {crypto_index}")
plt.xlabel("Time step")
plt.ylabel("Rank")
plt.show()


def plot_rank_distributions_over_time(rankings, save_dir, segment_size=100):
    num_segments = len(rankings) // segment_size
    num_cryptos = rankings.shape[1]

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        segment_rankings = rankings[start:end]

        plt.figure(figsize=(15, 10))
        plt.boxplot([segment_rankings[:, j] for j in range(num_cryptos)])
        plt.title(f"Distribution of Rankings for Each Cryptocurrency (Time steps {start} to {end})")
        plt.xlabel("Cryptocurrency")
        plt.ylabel("Rank")
        plt.ylim(0, num_cryptos)  # Set y-axis limits to the full range of possible ranks
        plt.tight_layout()

        # Save the figure in the specified directory
        save_path = os.path.join(save_dir, f"rank_distribution_segment_{i + 1}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"Created {num_segments} visualizations in {save_dir}")


def plot_rankings_line_over_time(rankings, save_dir, segment_size=500):
    num_segments = len(rankings) // segment_size
    num_cryptos = rankings.shape[1]

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_segments):
        start = i * segment_size
        end = (i + 1) * segment_size
        segment_rankings = rankings[start:end]

        plt.figure(figsize=(20, 10))
        for j in range(num_cryptos):
            plt.plot(range(start, end), segment_rankings[:, j], label=f'Crypto {j}')

        plt.title(f"Rankings for Each Cryptocurrency (Time steps {start} to {end})")
        plt.xlabel("Time step")
        plt.ylabel("Rank")
        plt.ylim(0, num_cryptos)  # Set y-axis limits to the full range of possible ranks
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()

        # Save the figure in the specified directory
        save_path = os.path.join(save_dir, f"rankings_line_segment_{i + 1}.png")
        plt.savefig(save_path)
        plt.close()

    print(f"Created {num_segments} line plot visualizations in {save_dir}")


# Line plot of ranking over time
save_directory = os.path.join(data_dir, results_dir, 'ranking_line_plots')
plot_rankings_line_over_time(rankings, save_directory, segment_size=20)

# Rank distribution box plot over time
save_directory = os.path.join(data_dir, results_dir, 'ranking_distribution')
plot_rank_distributions_over_time(rankings, save_directory, segment_size=20)
