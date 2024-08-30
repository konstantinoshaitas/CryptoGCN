import numpy as np
import pandas as pd
import os
import empyrical as ep

# Load Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
ASSET_NAMES_PATH = os.path.join(BASE_DIR, '..', 'data', 'correlation_data', 'aggregated_asset_data.csv')
ROLLING_RESULTS_DIR = os.path.join(RESULTS_DIR, 'rolling_window_results')
PLAY_RESULTS_DIR = os.path.join(RESULTS_DIR, 'play_results')

# Load asset names
df = pd.read_csv(ASSET_NAMES_PATH, parse_dates=['time'])
asset_names = df.columns[1:]  # Skipping the 'time' column

# Print the first asset name to verify
print("First Asset Name:", asset_names[0])

# Load numpy arrays
predictions = np.load(os.path.join(RESULTS_DIR, "predictions.npy"))
rankings = np.load(os.path.join(RESULTS_DIR, "rankings.npy"))
y_test = np.load(os.path.join(RESULTS_DIR, "y_test.npy"))

# Load numpy arrays for play data
play_predictions = np.load(os.path.join(PLAY_RESULTS_DIR, "predictions.npy"))
play_rankings = np.load(os.path.join(PLAY_RESULTS_DIR, "rankings.npy"))
play_y_test = np.load(os.path.join(PLAY_RESULTS_DIR, "y_test.npy"), allow_pickle=True)

# Load numpy arrays for all rolling windows
predictions_windows = []
rankings_windows = []
y_test_windows = []

i = 0
while True:
    try:
        predictions = np.load(os.path.join(ROLLING_RESULTS_DIR, f"predictions_roll_{i}.npy"))
        rankings = np.load(os.path.join(ROLLING_RESULTS_DIR, f"rankings_roll_{i}.npy"))
        y_test = np.load(os.path.join(ROLLING_RESULTS_DIR, f"y_test_roll_{i}.npy"))

        predictions_windows.append(predictions)
        rankings_windows.append(rankings)
        y_test_windows.append(y_test)

        i += 1
    except FileNotFoundError:
        print(f"Loaded {i} windows.")
        break

print(f"Number of windows loaded: {len(predictions_windows)}")


def long_short_portfolio_returns(predictions, y_test, rankings, asset_names, long_n=3, short_n=3, risk_free_rate=0.02,
                                 annualisation_factor=1095):
    """
    Evaluate the portfolio performance by going long on the top N assets and short on the bottom M assets.

    :param predictions: Predicted returns from the model.
    :param y_test: Actual returns.
    :param rankings: Rankings of the assets based on predictions.
    :param asset_names: Names of the assets corresponding to the columns in predictions and y_test.
    :param long_n: Number of top assets to go long on.
    :param short_n: Number of bottom assets to go short on.
    :param risk_free_rate: Annual risk-free rate for Sharpe Ratio calculation (e.g., 0.02 for 2%).
    :return: Portfolio returns, cumulative returns, Sortino ratio, Sharpe ratio.
    """
    # Initialize portfolio returns array
    portfolio_returns = []

    for i in range(predictions.shape[0]):
        long_return, short_return = 0, 0

        # If long_n is greater than 0, calculate long return
        if long_n > 0:
            long_positions = rankings[i, :long_n]
            long_return = np.mean(y_test[i, long_positions])
            long_assets = asset_names[long_positions]

        # If short_n is greater than 0, calculate short return
        if short_n > 0:
            short_positions = rankings[i, -short_n:]
            short_return = -np.mean(y_test[i, short_positions])
            short_assets = asset_names[short_positions]

        # Total portfolio return is the sum of long and short returns
        portfolio_return = long_return + short_return
        portfolio_returns.append(portfolio_return)

    portfolio_returns = np.array(portfolio_returns)

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1

    # Calculate Sortino ratio (risk-free rate assumed to be 0 for simplicity)
    sortino_ratio = ep.sortino_ratio(portfolio_returns, required_return=0, annualization=annualisation_factor)

    # Calculate Sharpe ratio
    sharpe_ratio = ep.sharpe_ratio(portfolio_returns, risk_free=risk_free_rate, annualization=annualisation_factor)

    return portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio


def long_only_benchmark(y_test, risk_free_rate=0.02, annualisation_factor=1095):
    """
    Evaluate the performance of a long-only portfolio on all assets based on actual returns.

    :param y_test: Actual returns.
    :param risk_free_rate: Annual risk-free rate for Sharpe Ratio calculation (e.g., 0.02 for 2%).
    :return: Benchmark portfolio returns, cumulative returns, Sortino ratio, Sharpe ratio.
    """
    # Average return across all assets for each time step
    portfolio_returns = np.mean(y_test, axis=1)

    # Calculate cumulative returns
    cumulative_returns = np.cumprod(1 + portfolio_returns) - 1

    # Calculate Sortino ratio (risk-free rate assumed to be 0 for simplicity)
    downside_risk = np.std(portfolio_returns[portfolio_returns < 0])
    sortino_ratio = ep.sortino_ratio(portfolio_returns, required_return=0, annualization=1095)

    # Calculate Sharpe Ratio
    sharpe_ratio = ep.sharpe_ratio(portfolio_returns, risk_free=risk_free_rate, annualization=annualisation_factor)

    return portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio


def evaluate_performance(predictions, y_test, rankings, asset_names, long_n=3, short_n=3, risk_free_rate=0.0):
    """Main function to evaluate performance."""
    # Evaluate long-short strategy
    portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio = long_short_portfolio_returns(
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

    # Evaluate long-only benchmark
    benchmark_returns, benchmark_cumulative, benchmark_sortino, benchmark_sharpe = long_only_benchmark(
        y_test, risk_free_rate=risk_free_rate)

    print(f"Long-Only Benchmark Sortino Ratio: {benchmark_sortino:.4f}")
    print(f"Long-Only Benchmark Sharpe Ratio: {benchmark_sharpe:.4f}")
    print(f"Long-Only Benchmark Final Cumulative Return: {benchmark_cumulative[-1]:.4f}")

    return (portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio, benchmark_returns, benchmark_cumulative,
            benchmark_sortino, benchmark_sharpe)


# Run evaluation with specified long and short asset counts
evaluate_performance(predictions, y_test, rankings, asset_names, long_n=2, short_n=0, risk_free_rate=0.00004455822024)

sortino_ratios = []
sharpe_ratios = []
final_cumulative_returns = []

# Variables to store benchmark metrics
benchmark_sortino_ratios = []
benchmark_sharpe_ratios = []
benchmark_final_cumulative_returns = []

# Loop through all rolling windows
for i in range(len(predictions_windows)):
    portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio, _, benchmark_cumulative, benchmark_sortino, benchmark_sharpe = evaluate_performance(
        predictions_windows[i], y_test_windows[i], rankings_windows[i], asset_names, long_n=3, short_n=0, risk_free_rate=0.00004455822024
    )
    sortino_ratios.append(sortino_ratio)
    sharpe_ratios.append(sharpe_ratio)
    final_cumulative_returns.append(cumulative_returns[-1])  # Store only the final cumulative return

    # Store benchmark metrics for each window
    benchmark_sortino_ratios.append(benchmark_sortino)
    benchmark_sharpe_ratios.append(benchmark_sharpe)
    benchmark_final_cumulative_returns.append(benchmark_cumulative[-1])

# Print average and standard deviation of the metrics across windows for the strategy
print(f"Average Sortino Ratio (Strategy): {np.mean(sortino_ratios):.4f}, Std Dev: {np.std(sortino_ratios):.4f}")
print(f"Average Sharpe Ratio (Strategy): {np.mean(sharpe_ratios):.4f}, Std Dev: {np.std(sharpe_ratios):.4f}")
print(f"Average Final Cumulative Return (Strategy): {np.mean(final_cumulative_returns):.4f}, Std Dev: {np.std(final_cumulative_returns):.4f}")

# Print average and standard deviation of the metrics across windows for the benchmark
print(f"Average Sortino Ratio (Benchmark): {np.mean(benchmark_sortino_ratios):.4f}, Std Dev: {np.std(benchmark_sortino_ratios):.4f}")
print(f"Average Sharpe Ratio (Benchmark): {np.mean(benchmark_sharpe_ratios):.4f}, Std Dev: {np.std(benchmark_sharpe_ratios):.4f}")
print(f"Average Final Cumulative Return (Benchmark): {np.mean(benchmark_final_cumulative_returns):.4f}, Std Dev: {np.std(benchmark_final_cumulative_returns):.4f}")

print(f'\n\n')
evaluate_performance(play_predictions, play_y_test, play_rankings, asset_names, long_n=1, short_n=4, risk_free_rate=0.00004455822024/2)