import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
import os

class CorrelationMatrix:
    def __init__(self, data, window_size=48):
        """
        Initializes the CorrelationMatrix with the data.

        :param data: A DataFrame containing data for multiple assets. Each column should represent an asset.
        :param window_size: The size of the sliding window for temporal correlation.
        """
        self.data = data.apply(pd.to_numeric, errors='coerce')
        self.window_size = window_size
        self.original_matrices = []
        self.denoised_matrices = []

        # Calculate number of assets (N)
        self.N = self.data.shape[1]

        # Set T to the window size
        self.T = window_size

        # Calculate q as N/T
        self.q = self.N / self.T

        # Placeholders for eigenvalues and eigenvectors
        self.eigenvalues = []
        self.eigenvectors = []

        # Marcenko-Pastur distribution parameters
        self.lambda_minus = (1 - np.sqrt(self.q)) ** 2
        self.lambda_plus = (1 + np.sqrt(self.q)) ** 2

    def calculate_returns(self):
        """
        Calculate standard returns for each asset.
        """
        self.returns = self.data.pct_change().dropna()

    def calculate_volatility(self):
        """
        Calculate standard deviation of returns and EWMA volatility for each asset.
        """
        self.volatility = self.returns.rolling(window=self.window_size).std()
        self.ewma_volatility = self.returns.pow(2).ewm(span=self.window_size).mean().pow(0.5)
        self.volatility = self.volatility.dropna()
        self.ewma_volatility = self.ewma_volatility.dropna()

    def compute_rolling_correlations(self):
        """
        Computes rolling correlation matrices for the data.
        """
        num_time_points = self.data.shape[0]
        window_size = self.window_size
        num_windows = num_time_points - window_size + 1

        for start in range(num_windows):
            end = start + window_size
            window_data = self.data.iloc[start:end]
            correlation_matrix = window_data.corr(method='pearson')
            self.original_matrices.append(correlation_matrix)

    def compute_eigenvalues_eigenvectors(self):
        """
        Computes eigenvalues and eigenvectors for each rolling correlation matrix.
        """
        for matrix in self.original_matrices:
            eigvals, eigvecs = np.linalg.eigh(matrix)
            self.eigenvalues.append(eigvals)
            self.eigenvectors.append(eigvecs)

    def denoise_correlation_matrices(self):
        """
        Denoises the correlation matrices using Random Matrix Theory while keeping the trace unchanged.
        """
        for i, eigvals in enumerate(self.eigenvalues):
            # Filter eigenvalues
            filtered_eigvals = np.where((eigvals >= self.lambda_minus) & (eigvals <= self.lambda_plus), 0, eigvals)

            # Reconstruct the denoised correlation matrix
            eigvecs = self.eigenvectors[i]
            denoised_matrix = eigvecs @ np.diag(filtered_eigvals) @ eigvecs.T

            # Ensure the matrix is symmetric
            denoised_matrix = (denoised_matrix + denoised_matrix.T) / 2

            # Adjust the trace to be equal to N (since it's a correlation matrix)
            np.fill_diagonal(denoised_matrix, 1)

            # Clip values to be within the range [-1, 1]
            denoised_matrix = np.clip(denoised_matrix, -1, 1)

            self.denoised_matrices.append(denoised_matrix)

    def plot_correlation_matrices(self, start, end):
        """
        Plots the original and denoised correlation matrices side by side for the specified range of indices.
        """
        for idx in range(start, end):
            if idx >= len(self.original_matrices) or idx >= len(self.denoised_matrices):
                print(f"Index {idx} is out of bounds. Skipping.")
                continue

            original_matrix = self.original_matrices[idx]
            denoised_matrix = self.denoised_matrices[idx]

            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            sns.heatmap(original_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[0],
                        xticklabels=self.data.columns, yticklabels=self.data.columns, vmin=-1, vmax=1)
            axes[0].set_title(f'Original Correlation Matrix (Index {idx})')

            sns.heatmap(denoised_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=axes[1],
                        xticklabels=self.data.columns, yticklabels=self.data.columns, vmin=-1, vmax=1)
            axes[1].set_title(f'Denoised Correlation Matrix (Index {idx})')

            plt.show()

    def plot_eigenvalue_distribution(self, start, end):
        """
        Plots the density distribution of eigenvalues and compares them with the Marcenko-Pastur distribution.
        """
        for idx in range(start, end):
            if idx >= len(self.eigenvalues):
                print(f"Index {idx} is out of bounds. Skipping.")
                continue

            eigvals = self.eigenvalues[idx]
            density = gaussian_kde(eigvals)
            x = np.linspace(min(eigvals), max(eigvals), 1000)
            y = density(x)

            # Marcenko-Pastur distribution
            mp_dist = lambda x: np.sqrt((self.lambda_plus - x) * (x - self.lambda_minus)) / (2 * np.pi * self.q * x) \
                if self.lambda_minus < x < self.lambda_plus else 0
            mp_y = np.array([mp_dist(val) for val in x])

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(eigvals, bins=50, density=True, alpha=0.6, color='b', label='Empirical Eigenvalues')
            ax.plot(x, y, color='blue', label='Empirical Eigenvalues Density')
            ax.plot(x, mp_y, color='red', linestyle='--', label='Marcenko-Pastur')
            ax.axvline(self.lambda_minus, color='yellow', linestyle='--', label='λ-')
            ax.axvline(self.lambda_plus, color='green', linestyle='--', label='λ+')

            ax.set_title(f'Eigenvalue Distribution for Matrix {idx + 1}')
            ax.set_xlabel('Eigenvalues')
            ax.set_ylabel('Density Distribution')
            ax.legend()

            plt.show()

# Test usage
if __name__ == "__main__":
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Construct the path to the data folder
    data_dir = os.path.join(script_dir, '..', 'data', 'correlation_data')

    # Load the aggregated asset data CSV
    data_path = os.path.join(data_dir, 'aggregated_asset_data.csv')
    data = pd.read_csv(data_path, index_col=0)

    # Initialize CorrelationMatrix object with a window size of 21
    corr_matrix = CorrelationMatrix(data, window_size=21)

    # Calculate returns and volatility
    corr_matrix.calculate_returns()
    corr_matrix.calculate_volatility()

    # Compute rolling correlations
    corr_matrix.compute_rolling_correlations()

    # Compute eigenvalues and eigenvectors
    corr_matrix.compute_eigenvalues_eigenvectors()

    # Index Range
    start = 0
    end = 3

    # Plot the eigenvalue distribution before denoising
    corr_matrix.plot_eigenvalue_distribution(start=start, end=end)

    # Denoise the correlation matrices
    corr_matrix.denoise_correlation_matrices()

    # Plot the correlation matrices for the specified range
    corr_matrix.plot_correlation_matrices(start=start, end=end)
