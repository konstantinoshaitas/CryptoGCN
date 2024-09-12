import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
import random
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import BayesianOptimization
from lstm_baseline import build_lstm_baseline

# Set seeds for reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Load Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGGREGATED_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'correlation_data', 'aggregated_asset_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results', 'comparative_results')
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
sys.path.append(MODELS_DIR)

from visualisations import plot_values_time, plot_predictions_vs_true

# Model Parameters
SEQUENCE_LENGTH = 21


def load_and_process_aggregated_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    df.sort_values(by='time', inplace=True)

    # Calculate t+1 returns for each asset
    for col in df.columns:
        if col != 'time':  # Skip the time column
            df[f'{col}_t+1_return'] = df[col].pct_change().shift(-1)

    return df[['time'] + [col for col in df.columns if '_t+1_return' in col]]


def prepare_dataset():
    df = load_and_process_aggregated_data(AGGREGATED_DATA_PATH)
    asset_names = df.columns[1:]  # Excluding the 'time' column
    train_index = int(len(df) * 0.7)
    valid_index = int(len(df) * 0.80)

    train_df = df.iloc[:train_index]
    valid_df = df.iloc[train_index:valid_index]
    test_df = df.iloc[valid_index:]

    def create_sequences_and_targets(data):
        sequences, targets, times = [], [], []
        for i in range(len(data) - SEQUENCE_LENGTH):
            seq = data.iloc[i:i + SEQUENCE_LENGTH, 1:].values
            target = data.iloc[i + SEQUENCE_LENGTH, 1:].values
            sequences.append(seq)
            targets.append(target)
            times.append(data.iloc[i + SEQUENCE_LENGTH]['time'])
        return np.array(sequences), np.array(targets), np.array(times)

    x_train, y_train, train_times = create_sequences_and_targets(train_df)
    x_valid, y_valid, validation_times = create_sequences_and_targets(valid_df)
    x_test, y_test, test_times = create_sequences_and_targets(test_df)

    return (x_train, y_train, train_times, x_valid, y_valid,
            validation_times, x_test, y_test, test_times, asset_names)


def main():
    batch_size = 16
    epochs = 500

    # Load and prepare the data
    (x_train, y_train, train_times, x_valid, y_valid, validation_times, x_test, y_test, test_times, asset_names) = prepare_dataset()

    # Convert data to float32 and ensure correct shape
    x_train = x_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_valid = y_valid.astype(np.float32)
    y_test = y_test.astype(np.float32)

    x_test = x_test[:-1]
    y_test = y_test[:-1]
    test_times = test_times[:-1]

    def build_tuning_model(hp):
        # Wrapper function that passes sequence_length and num_assets to build_lstm_baseline
        return build_lstm_baseline(hp, sequence_length=SEQUENCE_LENGTH, num_assets=x_train.shape[2])

    # Set up Keras Tuner with Bayesian Optimization
    tuner = BayesianOptimization(
        build_tuning_model,  # Wrapper function instead of build_lstm_baseline
        objective='val_loss',
        max_trials=20,  # How many different hyperparameter combinations to try
        executions_per_trial=5,  # Number of times to run each trial
        directory='bayesian_tuning',
        project_name='lstm_tuning'
    )

    # Search for the best hyperparameters
    tuner.search(
        x_train, y_train,
        validation_data=(x_valid, y_valid),
        epochs=100,
        batch_size=batch_size,
        callbacks=[EarlyStopping(monitor='val_loss', patience=15)]
    )

    # Get the best hyperparameters
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    # Build the best model and train
    model = tuner.hypermodel.build(best_hps)
    model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
              epochs=epochs, batch_size=batch_size,
              callbacks=[EarlyStopping(monitor='val_loss', patience=10)]
              )

    # Evaluate the model on the test set
    test_loss = model.evaluate(x_test, y_test)
    print(f"Test Loss: {test_loss:.4f}")

    # Generate predictions on the test set
    predictions = model.predict(x_test)
    rankings = np.argsort(-predictions, axis=1)

    # Save predictions and test data for later comparison
    np.save(os.path.join(RESULTS_DIR, "lstm_baseline_predictions.npy"), predictions)
    np.save(os.path.join(RESULTS_DIR, "lstm_baseline_y_test.npy"), y_test)
    np.save(os.path.join(RESULTS_DIR, "lstm_baseline_test_times.npy"), test_times)
    np.save(os.path.join(RESULTS_DIR, "lstm_baseline_rankings.npy"), rankings)

    # Plot predictions and true values
    plot_values_time(predictions, asset_names=asset_names, title_='Predictions', time_values=test_times)
    plot_values_time(y_test, asset_names=asset_names, title_='True Returns', time_values=test_times)

    asset0predictions = np.array(predictions[:, 10]).reshape(-1, 1)
    asset0true = np.array(y_test[:, 10]).reshape(-1, 1)
    asset_names_comparison = [asset_names[10]]

    plot_predictions_vs_true(asset0predictions, asset0true, asset_names=asset_names_comparison, time_values=test_times)


if __name__ == "__main__":
    main()
