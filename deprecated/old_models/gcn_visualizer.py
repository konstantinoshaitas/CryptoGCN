import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class GCNVisualizer:
    @staticmethod
    def visualize_graph(adjacency_matrix, num_nodes_to_show=15, threshold=0.5, save_path=None):
        """
        Visualize the graph structure used in the GCN.

        :param adjacency_matrix: The adjacency matrix of the graph
        :param num_nodes_to_show: Number of nodes to show in the visualization
        :param threshold: Minimum correlation value to show an edge
        :param save_path: Path to save the visualization. If None, the plot will be displayed.
        """
        # Create a graph from the adjacency matrix
        G = nx.from_numpy_array(adjacency_matrix[:num_nodes_to_show, :num_nodes_to_show])

        # Remove edges below the threshold
        for (u, v, d) in list(G.edges(data=True)):
            if abs(d['weight']) < threshold:
                G.remove_edge(u, v)

        # Set up the plot
        plt.figure(figsize=(12, 8))

        # Draw the graph
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='lightblue',
                node_size=500, font_size=10, font_weight='bold')

        # Draw edge weights
        edge_labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels={e: f'{w:.2f}' for e, w in edge_labels.items()})

        plt.title("GCN Graph Structure (Subset of Nodes)")
        plt.axis('off')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    @staticmethod
    def visualize_graph_over_time(adjacency_matrices, num_nodes_to_show=15, threshold=0.5, interval=10, save_path=None):
        """
        Visualize how the graph structure changes over time.

        :param adjacency_matrices: List of adjacency matrices over time
        :param num_nodes_to_show: Number of nodes to show in each visualization
        :param threshold: Minimum correlation value to show an edge
        :param interval: Number of time steps between visualizations
        :param save_path: Path to save the visualization. If None, the plots will be displayed.
        """
        for i in range(0, len(adjacency_matrices), interval):
            adj_matrix = adjacency_matrices[i]
            title = f"GCN Graph Structure at Time Step {i}"
            if save_path:
                current_save_path = f"{save_path}_time_{i}.png"
            else:
                current_save_path = None

            GCNVisualizer.visualize_graph(adj_matrix, num_nodes_to_show, threshold, current_save_path)
            if not save_path:
                plt.title(title)
                plt.show()
