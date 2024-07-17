import pandas as pd
import numpy as np
#test

class CorrelationMatrix:
    def __init__(self, data, window_size=48):
        """
        Initializes the CorrelationMatrix with the data.

        :param data: A DataFrame containing data for multiple assets. Each column should represent an asset.
        :param window_size: The size of the sliding window for temporal correlation.
        """
        self.data = data.drop(columns=['time']).apply(pd.to_numeric, errors='coerce')  # Ensure all data is numeric and drop the time column
        self.window_size = window_size

    def calculate_returns(self):
        """
        Calculate daily returns for each asset.
        """
        self.returns = self.data.pct_change(fill_method=None).dropna()

    def calculate_volatility(self):
        """
        Calculate standard deviation of returns and EWMA volatility for each asset.
        """
        self.volatility = self.returns.rolling(window=self.window_size).std()
        self.ewma_volatility = self.returns.pow(2).ewm(span=self.window_size).mean().pow(0.5)
        self.volatility = self.volatility.dropna()
        self.ewma_volatility = self.ewma_volatility.dropna()

    def calculate_denoised_correlation_matrices(self, method='returns'):
        if method == 'returns':
            data_series = self.returns
        elif method == 'volatility':
            data_series = self.volatility
        elif method == 'ewma_volatility':
            data_series = self.ewma_volatility
        else:
            raise ValueError("Method should be 'returns', 'volatility', or 'ewma_volatility'")

        denoised_matrices = []
        for start in range(0, len(data_series) - self.window_size + 1):
            window_data = data_series[start:start + self.window_size]

            # Normalize each column individually
            normalized_data = (window_data - window_data.mean(axis=0)) / window_data.std(axis=0)

            # Calculate the correlation matrix
            correlation_matrix = normalized_data.corr()

            # Check the diagonals
            if not np.allclose(np.diag(correlation_matrix), 1.0):
                print("Warning: Diagonals of the correlation matrix are not 1.")

            denoised_matrix = self.denoise_correlation_matrix(correlation_matrix)
            denoised_matrices.append(denoised_matrix)

        return denoised_matrices

    def denoise_correlation_matrix(self, correlation_matrix):
        """
        Denoise the correlation matrix using Eigenvector Clipping.

        :param correlation_matrix: The original correlation matrix.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(correlation_matrix)
        lambda_plus = (1 + np.sqrt(correlation_matrix.shape[0] / correlation_matrix.shape[1])) ** 2

        # Identify noisy eigenvalues
        k = np.sum(eigenvalues > lambda_plus)

        # Replace noisy eigenvalues
        for i in range(k, len(eigenvalues)):
            eigenvalues[i] = np.mean(eigenvalues[k:])
            # eigenvalues[i] = 0  # alternative choice for replacing noisy eigenvalues

        denoised_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return denoised_matrix
