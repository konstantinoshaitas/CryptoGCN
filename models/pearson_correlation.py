import pandas as pd
import numpy as np


class CorrelationMatrix:
    def __init__(self, data):
        """
        Initializes the CorrelationMatrix with the data.

        :param data: A DataFrame containing data for multiple assets. Each column should represent an asset.
        """
        self.data = data

    def calculate_returns(self):
        """
        Calculate daily returns for each asset.
        """
        self.returns = self.data.pct_change().dropna()

    def calculate_volatility(self):
        """
        Calculate standard deviation of returns and EWMA volatility for each asset.
        """
        self.volatility = self.returns.rolling(window=48).std()
        self.ewma_volatility = self.returns.pow(2).ewm(span=48).mean().pow(0.5)
        self.volatility = self.volatility.dropna()
        self.ewma_volatility = self.ewma_volatility.dropna()

    def calculate_correlation_matrix(self, method='returns'):
        """
        Calculate the Pearson correlation matrix.

        :param method: Method to calculate correlation ('returns', 'volatility', 'ewma_volatility')
        """
        if method == 'returns':
            data_series = self.returns
        elif method == 'volatility':
            data_series = self.volatility
        elif method == 'ewma_volatility':
            data_series = self.ewma_volatility
        else:
            raise ValueError("Method should be 'returns', 'volatility', or 'ewma_volatility'")

        # Normalize the time series
        normalized_data = (data_series - data_series.mean()) / data_series.std()
        correlation_matrix = normalized_data.corr()
        return correlation_matrix

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

        denoised_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
        return denoised_matrix
