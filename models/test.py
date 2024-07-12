import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pearson_correlation import CorrelationMatrix

# Load the aggregated asset data CSV
data_path = r'C:\Users\koko\Desktop\THESIS\CryptoGCN\data\aggregated_asset_data.csv'
data = pd.read_csv(data_path)

# Initialize the CorrelationMatrix class
correlation_matrix = CorrelationMatrix(data)

# Calculate returns
correlation_matrix.calculate_returns()
print("Returns:")
print(correlation_matrix.returns.head())

# Calculate volatility
correlation_matrix.calculate_volatility()
print("\nVolatility:")
print(correlation_matrix.volatility.head())

# Calculate denoised correlation matrices
denoised_matrices = correlation_matrix.calculate_denoised_correlation_matrices(method='returns')

# Test the number of correlation matrices
expected_num_matrices = len(correlation_matrix.returns) - correlation_matrix.window_size + 1
actual_num_matrices = len(denoised_matrices)
print(f"\nExpected Number of Correlation Matrices: {expected_num_matrices}")
print(f"Actual Number of Correlation Matrices: {actual_num_matrices}")
assert expected_num_matrices == actual_num_matrices, "The number of correlation matrices does not match the expected value."

# Define the range of matrices to visualize
start_index = 50  # Adjust this index to select different matrices
end_index = 52    # Adjust this index to select different matrices

# Print and visualize original and denoised correlation matrices
print("\nOriginal and Denoised Correlation Matrices:")
for i in range(start_index, end_index):
    start = i * (correlation_matrix.window_size)
    end = start + correlation_matrix.window_size
    window_data = correlation_matrix.returns[start:end]

    # Normalize each column individually
    normalized_data = (window_data - window_data.mean(axis=0)) / window_data.std(axis=0)

    # Calculate the original correlation matrix
    original_matrix = normalized_data.corr()

    # Print matrices
    print(f"Matrix {i + 1} - Original:")
    print(original_matrix)
    print(f"Matrix {i + 1} - Denoised:")
    print(denoised_matrices[i])

    # Visualize matrices
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(original_matrix, annot=True, ax=ax[0], vmin=-1, vmax=1, cmap='coolwarm')
    ax[0].set_title(f'Original Correlation Matrix {i + 1}')
    sns.heatmap(denoised_matrices[i], annot=True, ax=ax[1], vmin=-1, vmax=1, cmap='coolwarm')
    ax[1].set_title(f'Denoised Correlation Matrix {i + 1}')

    # Adjust the labels
    for axis in ax:
        axis.set_xticklabels(axis.get_xticklabels(), rotation=45, ha='right', fontsize=10)
        axis.set_yticklabels(axis.get_yticklabels(), rotation=0, fontsize=10)

    plt.tight_layout()
    plt.show()
