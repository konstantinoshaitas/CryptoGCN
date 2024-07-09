import pandas as pd
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

print("\nDenoised Correlation Matrices:")
print(f'shape: {denoised_matrices[0].shape}')
for i, matrix in enumerate(denoised_matrices[:3]):  # Print first 3 matrices for brevity
    print(f"Matrix {i+1}:")
    print(matrix)
