import numpy as np
import pandas as pd
import os
from evaluation_methods import (evaluate_performance, calculate_kendall_tau, calculate_spearman_rho,
                                top_k_metrics, plot_lists, calculate_ndcg)
import seaborn as sns

# Load Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')
ASSET_NAMES_PATH = os.path.join(BASE_DIR, '..', 'data', 'correlation_data', 'aggregated_asset_data.csv')
MAIN_MODEL_RESULTS_DIR = os.path.join(RESULTS_DIR, 'lstm_gcn_results')
COMPARATIVE_RESULTS_DIR = os.path.join(RESULTS_DIR, 'comparative_results')
ROLLING_RESULTS_DIR = os.path.join(RESULTS_DIR, 'rolling_window_results')
PLAY_RESULTS_DIR = os.path.join(RESULTS_DIR, 'play_results')
PLOT_RESULTS = os.path.join(RESULTS_DIR, 'plot_results')

# Load asset names
df = pd.read_csv(ASSET_NAMES_PATH, parse_dates=['time'])
asset_names = df.columns[1:]

# Model data
predictions = np.load(os.path.join(MAIN_MODEL_RESULTS_DIR, "predictions.npy"))
predicted_rankings = np.load(os.path.join(MAIN_MODEL_RESULTS_DIR, "rankings.npy"))
y_test = np.load(os.path.join(MAIN_MODEL_RESULTS_DIR, "y_test.npy"))
test_times = np.load(os.path.join(MAIN_MODEL_RESULTS_DIR, "test_times.npy"), allow_pickle=True)
true_rankings = np.argsort(-y_test, axis=1)

# LSTM baseline data
lstm_baseline_predictions = np.load(os.path.join(COMPARATIVE_RESULTS_DIR, "lstm_baseline_predictions.npy"))
lstm_baseline_predicted_rankings = np.load(os.path.join(COMPARATIVE_RESULTS_DIR, "lstm_baseline_rankings.npy"))
lstm_baseline_y_test = np.load(os.path.join(COMPARATIVE_RESULTS_DIR, "lstm_baseline_y_test.npy"))
lstm_baseline_test_times = np.load(os.path.join(COMPARATIVE_RESULTS_DIR, "lstm_baseline_test_times.npy"), allow_pickle=True)
lstm_baseline_true_rankings = np.argsort(-lstm_baseline_y_test, axis=1)

'''
FINANCIAL DATA - LSTM AND HYBRID
'''

(hybrid_portfolio_returns, hybrid_cumulative_returns, hybrid_sortino_ratio, hybrid_sharpe_ratio, hybrid_max_drawdown, hybrid_benchmark_returns,
 hybrid_benchmark_cumulative,
 hybrid_benchmark_sortino, hybrid_benchmark_sharpe, hybrid_benchmark_max_drawdown) = evaluate_performance(predictions, y_test, predicted_rankings, asset_names,
                                                                                                          long_n=2, short_n=0, risk_free_rate=0.00004455822024)

print(f'\n')
(lstm_portfolio_returns, lstm_cumulative_returns, lstm_sortino_ratio, lstm_sharpe_ratio, lstm_max_drawdown, lstm_benchmark_returns, lstm_benchmark_cumulative,
 lstm_benchmark_sortino, lstm_benchmark_sharpe, lstm_benchmark_max_drawdown) = evaluate_performance(lstm_baseline_predictions, lstm_baseline_y_test,
                                                                                                    lstm_baseline_predicted_rankings, asset_names,
                                                                                                    long_n=2, short_n=0, risk_free_rate=0.00004455822024)

plot_lists(hybrid_cumulative_returns * 100, lstm_cumulative_returns * 100, hybrid_benchmark_cumulative * 100,
           labels=["Hybrid Model", "LSTM Baseline", 'All Assets Benchmark'], title="Cumulative Returns Over Time", ylabel='% Cumulative Return',
           x_values=test_times)

long_values = [1, 2, 3, 5]

for long in long_values:
    (_, l_cum_ret_iter, _, _, _, _,
     l_bench_cumr,
     _, _, _) = evaluate_performance(lstm_baseline_predictions, lstm_baseline_y_test,
                                     lstm_baseline_predicted_rankings, asset_names,
                                     long_n=long, short_n=0, risk_free_rate=0.00004455822024)

    (_, hybrid_cumulative_returns, _, _, _, _,
     hybrid_benchmark_cumulative,
     _, _, _) = evaluate_performance(predictions, y_test, predicted_rankings, asset_names,
                                     long_n=long, short_n=0,
                                     risk_free_rate=0.00004455822024)

    plot_lists(hybrid_cumulative_returns * 100, hybrid_benchmark_cumulative * 100, l_cum_ret_iter * 100,
               labels=[f"Hybrid Model: Long {long} Assets", 'All Assets Benchmark', f"LSTM Baseline: Long {long} Assets"],
               title="",
               ylabel='% Cumulative Return',
               x_values=test_times,
               linewidth=1.1,
               figsize=(6, 4),
               save_path=os.path.join(PLOT_RESULTS, f'cum_return_long_{long}_assets'))

HYBRID_LIST = []
LSTM_LIST = []
LABELS = []

hybrid_colors = sns.color_palette("crest", n_colors=len(long_values))
lstm_colors = sns.color_palette("flare", n_colors=len(long_values))
benchmark_color = ["black"]
all_colors = []
for h, l in zip(hybrid_colors, lstm_colors):
    all_colors.append(h)
    all_colors.append(l)
all_colors.extend(benchmark_color)

for x in long_values:
    (_, l_cum_ret_iter, _, _, _, _,
     l_bench_cumr,
     _, _, _) = evaluate_performance(lstm_baseline_predictions, lstm_baseline_y_test,
                                     lstm_baseline_predicted_rankings, asset_names,
                                     long_n=x, short_n=0, risk_free_rate=0.00004455822024)

    (_, hybrid_cumulative_returns, _, _, _, _,
     _,
     _, _, _) = evaluate_performance(predictions, y_test, predicted_rankings, asset_names,
                                     long_n=x, short_n=0,
                                     risk_free_rate=0.00004455822024)

    HYBRID_LIST.append(hybrid_cumulative_returns*100)
    LSTM_LIST.append(l_cum_ret_iter*100)

    LABELS.append(f"Hybrid Model: Long {x} Assets")
    LABELS.append(f"LSTM Baseline: Long {x} Assets")

ALL_LINES = []
for i in range(len(HYBRID_LIST)):
    ALL_LINES.append(HYBRID_LIST[i])
    ALL_LINES.append(LSTM_LIST[i])

# Add the benchmark only once to the plot
ALL_LINES.append(hybrid_benchmark_cumulative * 100)
LABELS.append('All Assets Benchmark')

plot_lists(*ALL_LINES,
           labels=LABELS,
           title="Cumulative Returns Over Time",
           ylabel='% Cumulative Return',
           x_values=test_times,
           custom_palette=all_colors,
           save_path=os.path.join(PLOT_RESULTS, 'cum_return_all_long_assets'))

'''
KENDALL TAU AND SPEARMAN RHO
'''

hybrid_model_tau_list, hybrid_model_tau_average = calculate_kendall_tau(predicted_rankings, true_rankings)
hybrid_model_rho_list, hybrid_model_rho_average = calculate_spearman_rho(predicted_rankings, true_rankings)
lstm_model_tau_list, lstm_model_tau_average = calculate_kendall_tau(lstm_baseline_predicted_rankings, lstm_baseline_true_rankings)
lstm_model_rho_list, lstm_model_rho_average = calculate_spearman_rho(lstm_baseline_predicted_rankings, lstm_baseline_true_rankings)

print(f'\nHybrid Model\nKendall T avg: {hybrid_model_tau_average:.3f}, Spearman Rho avg:{hybrid_model_rho_average:.3f}')
print(f'\nLSTM Base Model\nKendall T avg: {lstm_model_tau_average:.3f}, Spearman Rho avg:{lstm_model_rho_average:.3f}')


def moving_average(data, win_size):
    return np.convolve(data, np.ones(win_size) / win_size, mode='valid')


plot_lists(hybrid_model_tau_list, lstm_model_tau_list,
           labels=["Hybrid Kendall's Tau", "LSTM Kendall's Tau"],
           title="Kendall Rank Correlations Over Time",
           save_path=os.path.join(PLOT_RESULTS, 'tau_1'))

plot_lists(hybrid_model_rho_list, lstm_model_rho_list,
           labels=["Hybrid Spearman's Rho", "LSTM Spearman's Rho"],
           title="Spearman Rank Correlations Over Time",
           save_path=os.path.join(PLOT_RESULTS, 'rho_1'))

window_size = 120
hybrid_rho_smooth = moving_average(hybrid_model_rho_list, window_size)
hybrid_tau_smooth = moving_average(hybrid_model_tau_list, window_size)
lstm_rho_smooth = moving_average(lstm_model_rho_list, window_size)
lstm_tau_smooth = moving_average(lstm_model_tau_list, window_size)

plot_lists(hybrid_tau_smooth, lstm_tau_smooth,
           labels=["Hybrid Kendall's Tau", "LSTM Kendall's Tau"],
           title="Kendall Rank Correlations Over Time",
           save_path=os.path.join(PLOT_RESULTS, 'tau_smooth'))

plot_lists(hybrid_rho_smooth, lstm_rho_smooth,
           labels=[f"Hybrid Spearman's Rho {window_size} MA", f"LSTM Spearman's Rho {window_size} MA"],
           title="Spearman Rank Correlations Over Time",
           save_path=os.path.join(PLOT_RESULTS, 'rho_smooth'))

'''
TOP K ACCURACY AND PRECISION
'''

print(f'\n')
hybrid_avg_k_acc, hybrid_k_prec, hybrid_k = top_k_metrics(predicted_rankings,
                                                          true_rankings, k=4, top_a=10)
lstm_avg_k_acc, lstm_k_prec, lstm_k = top_k_metrics(lstm_baseline_predicted_rankings,
                                                    lstm_baseline_true_rankings, k=4, top_a=10)

print(f'Hybrid Model: avg top {hybrid_k} accuracy: {hybrid_avg_k_acc:.3f}, precision: {hybrid_k_prec:.3f}')
print(f'LSTM Model: avg top {lstm_k} accuracy: {lstm_avg_k_acc:.3f}, precision: {lstm_k_prec:.3f}')

hybrid_accuracies = []
hybrid_precisions = []
lstm_accuracies = []
lstm_precisions = []
a_value = 10
k_values = list(range(1, 8))

# Loop through the k values to get the Top-K Accuracy and Precision for both models
for k in k_values:
    hybrid_avg_k_acc, hybrid_k_prec, _ = top_k_metrics(predicted_rankings, true_rankings, k=k, top_a=a_value)
    lstm_avg_k_acc, lstm_k_prec, _ = top_k_metrics(lstm_baseline_predicted_rankings, lstm_baseline_true_rankings, k=k, top_a=a_value)

    hybrid_accuracies.append(hybrid_avg_k_acc)
    hybrid_precisions.append(hybrid_k_prec)
    lstm_accuracies.append(lstm_avg_k_acc)
    lstm_precisions.append(lstm_k_prec)

plot_lists(hybrid_precisions, lstm_precisions,
           labels=[f'Hybrid Model Precision (a={a_value})', f'LSTM Precision (a={a_value})'],
           title=f'Top-K Precision Score with (a={a_value}) for Hybrid and LSTM Models',
           xlabel='Top-K', ylabel='Precision Value', x_values=k_values,
           save_path=os.path.join(PLOT_RESULTS, 'K_precision'))

plot_lists(hybrid_accuracies, lstm_accuracies,
           labels=['Hybrid Model Top-K Accuracy', 'LSTM Top-K Accuracy'],
           title='Top-K Accuracy for Hybrid and LSTM Models',
           xlabel='Values of K', ylabel='Accuracy', x_values=k_values,
           save_path=os.path.join(PLOT_RESULTS, 'K_accuracy'))

'''
NDCG@K
'''
hybrid_ndcg_list, hybrid_avg_ndcg = calculate_ndcg(predicted_rankings, true_rankings, k=5)
lstm_ndcg_list, lstm_avg_ndcg = calculate_ndcg(lstm_baseline_predicted_rankings, lstm_baseline_true_rankings, k=5)

print(f'Hybrid NDCG@K = {hybrid_avg_ndcg}')
print(f'LSTM NDCG@K = {lstm_avg_ndcg}')

window_size = 120
hybrid_ndcg_smooth = moving_average(hybrid_ndcg_list, window_size)
lstm_ndcg_smooth = moving_average(lstm_ndcg_list, window_size)

plot_lists(hybrid_ndcg_smooth, lstm_ndcg_smooth,
           labels=["Hybrid NDCG", "LSTM NDCG"],
           title=f"NDCG {window_size} MOVING AVERAGE", ylabel='NDCG',
           save_path=os.path.join(PLOT_RESULTS, 'ndcg_smooth'))

h_ndcg_at_k = []
l_ndcg_at_k = []
for k in k_values:
    _, h_ = calculate_ndcg(predicted_rankings, true_rankings, k=k)
    _, l_ = calculate_ndcg(lstm_baseline_predicted_rankings, lstm_baseline_true_rankings, k=k)
    h_ndcg_at_k.append(h_)
    l_ndcg_at_k.append(l_)

plot_lists(h_ndcg_at_k, l_ndcg_at_k,
           labels=["Hybrid NDCG", "LSTM NDCG"],
           title=f"NDCG AVERAGE @K", ylabel='NDCG',
           save_path=os.path.join(PLOT_RESULTS, 'ndcg_at_k'))

'''
ROLLING WINDOW DATA
'''
'''
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
'''
'''
ROLLING WINDOW EVAL
'''
'''
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
print(f"Average Final Cumulative Return (Benchmark): {np.mean(benchmark_final_cumulative_returns):.4f}, "
      f"Std Dev: {np.std(benchmark_final_cumulative_returns):.4f}")
'''
