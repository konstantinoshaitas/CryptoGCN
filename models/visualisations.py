import matplotlib.pyplot as plt
import networkx as nx
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PLOTS_DIR = os.path.join(BASE_DIR, '..', 'plots')


def plot_values_time(return_values, asset_names=None, title_='Predictions', time_values=None):
    """
    Plots values for each asset over time

    :param return_values: np.array of shape (number of time steps, number of assets) containing predictions or true values
    :param asset_names: Optional list of asset names
    :param title_: Title for the plot
    :param time_values: Optional array of time values to use as the x-axis labels
    """
    num_assets = return_values.shape[1]
    plt.figure(figsize=(14, 7))

    for i in range(num_assets):
        # Use time_values for x-axis if provided, otherwise just use indices as time steps
        plt.plot(time_values if time_values is not None else range(return_values.shape[0]),
                 return_values[:, i],
                 label=asset_names[i] if asset_names is not None else f'Asset {i + 1}')

    plt.title(f'{title_} for Each Asset Over Time')
    plt.xlabel('Time' if time_values is not None else 'Time Step')
    plt.ylabel(f'{title_} Value')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def plot_predictions_vs_true(predicted_values, true_values, asset_names=None, title_='Predictions vs True Values', time_values=None):
    """
    Plots predicted and true values for each asset over time

    :param predicted_values: np.array of shape (number of time steps, number of assets) containing predicted values
    :param true_values: np.array of shape (number of time steps, number of assets) containing true return values
    :param asset_names: Optional list of asset names
    :param title_: Title for the plot
    :param time_values: Optional array of time values to use as the x-axis labels
    """
    num_assets = predicted_values.shape[1]
    plt.figure(figsize=(14, 7))

    for i in range(num_assets):
        # Use time_values for x-axis if provided, otherwise just use indices as time steps
        time_axis = time_values if time_values is not None else range(predicted_values.shape[0])

        # Plot predicted values
        plt.plot(time_axis, predicted_values[:, i], label=f'Predicted {asset_names[i]}' if asset_names is not None else f'Predicted Asset {i + 1}',
                 linestyle='--')

        # Plot true values
        plt.plot(time_axis, true_values[:, i], label=f'True {asset_names[i]}' if asset_names is not None else f'True Asset {i + 1}', linestyle='-')

    plt.title(f'{title_} for Each Asset Over Time')
    plt.xlabel('Time' if time_values is not None else 'Time Step')
    plt.ylabel(f'{title_} Value')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()


def main():
    G = nx.random_geometric_graph(9, 1)
    pos = nx.spring_layout(G)  # Positions nodes using the spring layout algorithm
    plt.figure(figsize=(8, 6))
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color='olive', edgecolors='black', linewidths=1.75)
    nx.draw_networkx_edges(G, pos, width=1.5, edge_color='grey')
    # nx.draw_networkx_labels(G, pos, font_size=12, font_family="sans-serif", font_weight="bold")
    plt.axis('off')
    plt.savefig(os.path.join(PLOTS_DIR, "networkx_graph.png"), format="PNG", transparent=True)
    print(f'networkx_graph.png saved in {PLOTS_DIR}')
    plt.show()


if __name__ == "__main__":
    main()
