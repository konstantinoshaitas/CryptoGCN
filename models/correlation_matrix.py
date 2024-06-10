import numpy as np
import pandas as pd
from scipy.linalg import eigh
import ace_tools as tools

# Placeholder: Load the data
# Assuming the data is in a CSV file with columns: 'Date', 'Cryptocurrency', 'Price'
# Adjust the file path and structure as necessary
data = pd.read_csv('path_to_your_data.csv')

# Pivot the data to have cryptocurrencies as columns and dates as rows
data_pivot = data.pivot(index='Date', columns='Cryptocurrency', values='Price')

# Calculate standard returns (not log returns)
returns = data_pivot.pct_change().dropna()

# Calculate the correlation matrix
C = np.corrcoef(returns, rowvar=False)

# Perform eigenvalue decomposition
eigenvalues, eigenvectors = eigh(C)

# Sort eigenvalues and corresponding eigenvectors in descending order
sorted_indices = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[sorted_indices]
eigenvectors = eigenvectors[:, sorted_indices]

# Determine the maximum eigenvalue threshold (Marcenko-Pastur distribution can be used here)
T, N = returns.shape
lambda_plus = (1 + np.sqrt(N / T)) ** 2

# Denoise the correlation matrix
k = np.sum(eigenvalues > lambda_plus)
lambda_star = np.zeros_like(eigenvalues)
lambda_star[:k] = eigenvalues[:k]

# Construct the denoised correlation matrix
C_denoised = np.dot(eigenvectors, np.dot(np.diag(lambda_star), eigenvectors.T))

# Display the denoised correlation matrix
C_denoised_df = pd.DataFrame(C_denoised, index=data_pivot.columns, columns=data_pivot.columns)
tools.display_dataframe_to_user(name="Denoised Correlation Matrix", dataframe=C_denoised_df)
C_denoised_df
