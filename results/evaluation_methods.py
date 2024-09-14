import numpy as np
import pandas as pd
import os
import empyrical as ep
from scipy.stats import spearmanr, kendalltau
import matplotlib.pyplot as plt
import empyrical as ep
from sklearn.metrics import ndcg_score


def long_short_portfolio_returns(predictions, y_test, rankings, asset_names, long_n=3, short_n=3, risk_free_rate=0.02,
                                 annualisation_factor=1095):
    """
    Evaluate the portfolio performance by going long on the top N assets and short on the bottom M assets.
    """
    portfolio_returns = []

    for i in range(predictions.shape[0]):
        long_return, short_return = 0, 0

        # Calculate long return
        if long_n > 0:
            long_positions = rankings[i, :long_n]
            long_return = np.mean(y_test[i, long_positions])

        # Calculate short return
        if short_n > 0:
            short_positions = rankings[i, -short_n:]
            short_return = -np.mean(y_test[i, short_positions])

        portfolio_return = long_return + short_return
        portfolio_returns.append(portfolio_return)

    portfolio_returns = np.array(portfolio_returns)

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1

    # Calculate max drawdown
    max_drawdown = ep.max_drawdown(portfolio_returns)

    # Calculate Sortino ratio
    sortino_ratio = ep.sortino_ratio(portfolio_returns, required_return=0, annualization=annualisation_factor)

    # Calculate Sharpe ratio
    sharpe_ratio = ep.sharpe_ratio(portfolio_returns, risk_free=risk_free_rate, annualization=annualisation_factor)

    return portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio, max_drawdown


def long_only_benchmark(y_test, risk_free_rate=0.02, annualisation_factor=1095):
    """
    Evaluate the performance of a long-only portfolio on all assets based on actual returns.
    """
    portfolio_returns = np.mean(y_test, axis=1)

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1

    # Calculate max drawdown
    max_drawdown = ep.max_drawdown(portfolio_returns)

    # Calculate Sortino ratio
    sortino_ratio = ep.sortino_ratio(portfolio_returns, required_return=0, annualization=annualisation_factor)

    # Calculate Sharpe ratio
    sharpe_ratio = ep.sharpe_ratio(portfolio_returns, risk_free=risk_free_rate, annualization=annualisation_factor)

    return portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio, max_drawdown


def evaluate_performance(predictions, y_test, rankings, asset_names, long_n=3, short_n=3, risk_free_rate=0.0):
    """
    Main function to evaluate performance of a strategy, including max drawdown.
    """
    # Evaluate long-short strategy
    portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio, max_drawdown = long_short_portfolio_returns(
        predictions, y_test, rankings, asset_names, long_n=long_n, short_n=short_n, risk_free_rate=risk_free_rate,
        annualisation_factor=1095)

    if long_n == 0:
        portfolio_type = f'Short-Only. Worst {short_n} Assets Portfolio'
    elif short_n == 0:
        portfolio_type = f'Long-Only. Best {long_n} Assets Portfolio'
    else:
        portfolio_type = f'Long-Short Portfolio {long_n} asset(s) long {short_n} asset(s) short'

    print(f"{portfolio_type} Strategy Sortino Ratio: {sortino_ratio:.4f}")
    print(f"{portfolio_type} Strategy Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"{portfolio_type} Strategy Final Cumulative Return: {cumulative_returns[-1]:.4f}")
    print(f"{portfolio_type} Strategy Max Drawdown: {max_drawdown:.4f}")

    # Evaluate long-only benchmark
    benchmark_returns, benchmark_cumulative, benchmark_sortino, benchmark_sharpe, benchmark_max_drawdown = long_only_benchmark(
        y_test, risk_free_rate=risk_free_rate)

    print(f"Long-Only Benchmark Sortino Ratio: {benchmark_sortino:.4f}")
    print(f"Long-Only Benchmark Sharpe Ratio: {benchmark_sharpe:.4f}")
    print(f"Long-Only Benchmark Final Cumulative Return: {benchmark_cumulative[-1]:.4f}")
    print(f"Long-Only Benchmark Max Drawdown: {benchmark_max_drawdown:.4f}")

    return (portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio, max_drawdown,
            benchmark_returns, benchmark_cumulative, benchmark_sortino, benchmark_sharpe, benchmark_max_drawdown)


def calculate_spearman_rho(predicted_rankings, true_rankings):
    """
    Calculate the Spearman's Rho (rank correlation) for each time step using precomputed rankings.

    :param predicted_rankings: Predicted rankings from the model.
    :param true_rankings: True rankings based on actual returns.
    :return: Average Spearman's Rho across all time steps.
    """
    rho_values = []
    for i in range(predicted_rankings.shape[0]):
        rho, _ = spearmanr(predicted_rankings[i], true_rankings[i])  # Directly use rankings
        rho_values.append(rho)

    return rho_values, np.mean(rho_values)


def calculate_kendall_tau(predicted_rankings, true_rankings):
    """
    Calculate the Kendall's Tau (rank correlation) for each time step using precomputed rankings.

    :param predicted_rankings: Predicted rankings from the model.
    :param true_rankings: True rankings based on actual returns.
    :return: Average Kendall's Tau across all time steps.
    """
    tau_values = []
    for i in range(predicted_rankings.shape[0]):
        tau, _ = kendalltau(predicted_rankings[i], true_rankings[i])  # Directly use rankings
        tau_values.append(tau)

    return tau_values, np.mean(tau_values)


def top_k_metrics(predicted_rankings, true_rankings, k=3, top_a=7):
    """
    Calculate Top-K accuracy and Precision for the top-K predicted assets.
    Precision is calculated against the top `a` true assets.

    :param predicted_rankings: Predicted rankings of assets (array shape: [time steps, assets]).
    :param true_rankings: True rankings of assets (array shape: [time steps, assets]).
    :param k: Number of top predicted assets to evaluate.
    :param top_a: Number of top true assets (in the larger pool) to evaluate precision.
    :return: Tuple containing (Top-K Accuracy, Precision) across all time steps.
    """
    total_top_k_accuracy = 0
    total_precision = 0
    total_steps = predicted_rankings.shape[0]

    for i in range(total_steps):
        pred_top_k = set(predicted_rankings[i, :k])  # Top K predicted assets
        true_top_k = set(true_rankings[i, :k])  # Top K true assets for accuracy
        true_top_a = set(true_rankings[i, :top_a])  # Top A true assets for precision

        # Number of true positives (correctly predicted top-K)
        true_positives_in_top_k = len(pred_top_k.intersection(true_top_k))  # For Top-K accuracy
        true_positives_in_top_a = len(pred_top_k.intersection(true_top_a))  # For Precision

        # Top-K Accuracy: Measures how many of the predicted top-K match the true top-K
        top_k_accuracy = true_positives_in_top_k / k
        total_top_k_accuracy += top_k_accuracy

        # Precision: True Positives / Predicted Positives (which is k)
        precision = true_positives_in_top_a / k
        total_precision += precision

    # Average over all time steps
    avg_top_k_accuracy = total_top_k_accuracy / total_steps
    avg_precision = total_precision / total_steps

    return avg_top_k_accuracy, avg_precision, k


def calculate_ndcg(predicted_rankings, true_rankings, k):
    """
    Calculate NDCG@k across all time steps using rankings.

    :param predicted_rankings: Predicted rankings (2D array: time_steps, assets).
    :param true_rankings: True rankings (2D array: time_steps, assets).
    :param k: Rank cutoff.
    :return: List of NDCG@k for each time step, and the average NDCG@k across all time steps.
    """
    ndcg_values = []

    # Loop over time steps (i.e., 757 iterations)
    for i in range(predicted_rankings.shape[0]):
        # Extract rankings for this time step (15 assets)
        pred_ranks = predicted_rankings[i]
        true_ranks = true_rankings[i]

        # Calculate NDCG for this time step
        ndcg_value = ndcg_score([true_ranks], [pred_ranks], k=k)
        ndcg_values.append(ndcg_value)

    # Return the list of NDCG values for each time step and the average NDCG across all time steps
    return ndcg_values, np.mean(ndcg_values)


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_lists(*lists, labels=None, title="Plot of Values", xlabel="Time", ylabel="Values", x_values=None,
               figsize=(12, 6), style="dark", custom_palette=None, palette="twilight", linewidth=2, alpha=1,
               marker=None, markersize=6, legend_loc='best', save_path=None):
    """
    Plot one or more lists or numpy arrays as lines on the same plot with enhanced aesthetics using Seaborn.

    :param lists: One or more lists or numpy arrays to plot.
    :param labels: Labels for each list, passed as a list of strings. Must match the number of lists provided.
    :param title: Title of the plot.
    :param xlabel: Label for the x-axis.
    :param ylabel: Label for the y-axis.
    :param x_values: Values for the X axis. If None, defaults to the index of the data.
    :param figsize: Size of the figure as a tuple (width, height).
    :param style: Seaborn style (e.g., 'darkgrid', 'whitegrid', 'dark', 'white', 'ticks').
    :param custom_palette: Used if you want to define multiple palettes, i.e. custom.
    :param palette: Color palette to use (e.g., 'deep', 'muted', 'pastel', 'bright', 'dark', 'colorblind').
    :param linewidth: Width of the plotted lines.
    :param alpha: Transparency of the lines (0 to 1).
    :param marker: Marker style for data points.
    :param markersize: Size of markers.
    :param legend_loc: Location of the legend (e.g., 'best', 'upper left', 'lower right').
    :param save_path: Path to save the figure. If None, the plot is displayed but not saved.
    """
    # Set the Seaborn style
    sns.set_style(style)

    # Create a new figure
    plt.figure(figsize=figsize)

    # Get color palette
    if custom_palette:
        colors = custom_palette
    else:
        colors = sns.color_palette(palette, n_colors=len(lists))

    for i, data in enumerate(lists):
        if x_values is not None:
            sns.lineplot(x=x_values, y=data, label=labels[i] if labels else None,
                         color=colors[i], linewidth=linewidth, alpha=alpha, marker=marker, markersize=markersize)
        else:
            sns.lineplot(x=range(len(data)), y=data, label=labels[i] if labels else None,
                         color=colors[i], linewidth=linewidth, alpha=alpha, marker=marker, markersize=markersize)

    # Customize the plot
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, linestyle='-', alpha=0.5)

    # Customize tick labels
    plt.tick_params(axis='both', which='major', labelsize=10)

    # If labels are provided, show the legend
    if labels is not None:
        plt.legend(loc=legend_loc, fontsize=10, frameon=True, framealpha=0.8)

    # Adjust layout to prevent clipping of labels
    plt.tight_layout()

    # Save the figure if a save path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    # Show the plot
    plt.show()
