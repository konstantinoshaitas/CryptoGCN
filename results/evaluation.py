import numpy as np
import pandas as pd
import os
import empyrical as ep

# Load Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
ASSET_NAMES_PATH = os.path.join(BASE_DIR, '..', 'data', 'correlation_data', 'aggregated_asset_data.csv')

# Load asset names
df = pd.read_csv(ASSET_NAMES_PATH, parse_dates=['time'])
asset_names = df.columns[1:]  # Skipping the 'time' column

# Print the first asset name to verify
print("First Asset Name:", asset_names[0])

# Load numpy arrays
predictions = np.load(os.path.join(RESULTS_DIR, "predictions.npy"))
rankings = np.load(os.path.join(RESULTS_DIR, "rankings.npy"))
y_test = np.load(os.path.join(RESULTS_DIR, "y_test.npy"))

print(predictions.shape)


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
            print(f"Time Step {i}: Long {long_assets.values}")

        # If short_n is greater than 0, calculate short return
        if short_n > 0:
            short_positions = rankings[i, -short_n:]
            short_return = -np.mean(y_test[i, short_positions])
            short_assets = asset_names[short_positions]
            print(f"Time Step {i}: Short {short_assets.values}")

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

    # Optional: Save the evaluation results
    # np.save(os.path.join(RESULTS_DIR, "portfolio_returns.npy"), portfolio_returns)
    # np.save(os.path.join(RESULTS_DIR, "cumulative_returns.npy"), cumulative_returns)
    # np.save(os.path.join(RESULTS_DIR, "benchmark_returns.npy"), benchmark_returns)
    # np.save(os.path.join(RESULTS_DIR, "benchmark_cumulative.npy"), benchmark_cumulative)

    return (portfolio_returns, cumulative_returns, sortino_ratio, sharpe_ratio, benchmark_returns, benchmark_cumulative,
            benchmark_sortino, benchmark_sharpe)


# Run evaluation with specified long and short asset counts
evaluate_performance(predictions, y_test, rankings, asset_names, long_n=3, short_n=0, risk_free_rate=0.00004455822024)
