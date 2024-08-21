import matplotlib.pyplot as plt


def plot_values_time(return_values, asset_names=None, title_='Predictions'):
    """
    Plots values for each asset over time

    :param title_: X values over time
    :param return_values: np.array of shape (number of time steps, number of assets) containing predictions or true vals
    :param asset_names: Optional list of asset names
    """
    num_assets = return_values.shape[1]
    plt.figure(figsize=(14, 7))

    for i in range(num_assets):
        plt.plot(return_values[:, i], label=asset_names[i] if asset_names is not None else f'Asset {i + 1}')

    plt.title(f'{title_} for Each Asset Over Time')
    plt.xlabel('Time Step')
    plt.ylabel(f'{title_} Value')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()
