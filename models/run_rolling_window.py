import os
import pandas as pd
import numpy as np
import tensorflow as tf
from Crypto_LSTM_GCN import EndToEndCryptoModel
from pearson_correlation import CorrelationMatrix
from visualisations import plot_values_time
import random

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Load Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed_sequential_data')
AGGREGATED_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'correlation_data', 'aggregated_asset_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results')

# Model Parameters
SEQUENCE_LENGTH = 21
DATA_POINTS_PER_DAY = 3  # 8-hourly data
TRAIN_DAYS = 600  # Approximately 2 Y
VALIDATION_DAYS = 60  # 2 Months
TEST_DAYS = 60  # 2 Month
STEP_DAYS = 120  # Move window by 4M each time

# Convert days to data points
TRAIN_SIZE = TRAIN_DAYS * DATA_POINTS_PER_DAY
VALIDATION_SIZE = VALIDATION_DAYS * DATA_POINTS_PER_DAY
TEST_SIZE = TEST_DAYS * DATA_POINTS_PER_DAY
STEP_SIZE = STEP_DAYS * DATA_POINTS_PER_DAY


def load_and_process_aggregated_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    print("Column names in the dataset:", df.columns)
    df.sort_values(by='time', inplace=True)

    for col in df.columns:
        if col != 'time':
            df[f'{col}_t+1_return'] = df[col].pct_change().shift(-1)

    return df[['time'] + [col for col in df.columns if '_t+1_return' in col]]


def create_sequences_and_targets(data):
    sequences = []
    targets = []
    for i in range(len(data) - SEQUENCE_LENGTH):
        seq = data.iloc[i:i + SEQUENCE_LENGTH, 1:].values.astype(np.float32)
        target = data.iloc[i + SEQUENCE_LENGTH, 1:].values.astype(np.float32)
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)


def prepare_rolling_window_dataset(df):
    total_windows = (len(df) - TRAIN_SIZE - VALIDATION_SIZE - TEST_SIZE) // STEP_SIZE + 1
    all_train_data = []
    all_validation_data = []
    all_test_data = []

    for i in range(total_windows):
        start_idx = i * STEP_SIZE
        train_end_idx = start_idx + TRAIN_SIZE
        validation_end_idx = train_end_idx + VALIDATION_SIZE
        test_end_idx = validation_end_idx + TEST_SIZE

        train_df = df.iloc[start_idx:train_end_idx]
        validation_df = df.iloc[train_end_idx:validation_end_idx]
        test_df = df.iloc[validation_end_idx:test_end_idx]

        x_train, y_train = create_sequences_and_targets(train_df)
        x_validation, y_validation = create_sequences_and_targets(validation_df)
        x_test, y_test = create_sequences_and_targets(test_df)

        all_train_data.append((x_train, y_train))
        all_validation_data.append((x_validation, y_validation))
        all_test_data.append((x_test, y_test))

    return all_train_data, all_validation_data, all_test_data


def main():
    # Load and prepare data
    df = load_and_process_aggregated_data(AGGREGATED_DATA_PATH)
    asset_names = df.columns[1:]  # Assuming first column is 'time' and the rest are assets

    all_train_data, all_validation_data, all_test_data = prepare_rolling_window_dataset(df)

    # Initialize the correlation matrix
    correlation_matrix = CorrelationMatrix(df.drop('time', axis=1), window_size=SEQUENCE_LENGTH)

    for window_index, ((x_train, y_train), (x_validation, y_validation), (x_test, y_test)) in enumerate(
            zip(all_train_data, all_validation_data, all_test_data)):
        print(f"\nProcessing window {window_index + 1}/{len(all_train_data)}")

        # Generate denoised correlation matrices for this window
        train_denoised, validation_denoised, test_denoised = correlation_matrix.run()

        # Ensure we have the correct number of correlation matrices
        train_adj_matrices = np.array(train_denoised[:len(x_train)], dtype=np.float32)
        validation_adj_matrices = np.array(validation_denoised[:len(x_validation)], dtype=np.float32)
        test_adj_matrices = np.array(test_denoised[:len(x_test)], dtype=np.float32)

        # Define and compile the model
        model = EndToEndCryptoModel(sequence_length=SEQUENCE_LENGTH, lstm_units=5, num_assets=x_train.shape[2], alpha=1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000003))

        # Train the model
        batch_size = 64
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, train_adj_matrices, y_train))
        train_dataset = train_dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

        validation_dataset = tf.data.Dataset.from_tensor_slices((x_validation, validation_adj_matrices, y_validation))
        validation_dataset = validation_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = len(y_train) // batch_size
        epochs = 6
        best_val_loss = float('inf')
        patience = 2
        wait = 0

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            for step, (x_batch, adj_batch, y_batch) in enumerate(train_dataset.take(steps_per_epoch)):
                loss = model.train_step((x_batch, adj_batch, y_batch))
                if step % 10 == 0:
                    print(f"Step {step}: Loss = {loss['loss']:.4f}")

            # Validation
            val_losses = []
            for x_batch, adj_batch, y_batch in validation_dataset:
                val_result = model.test_step((x_batch, adj_batch, y_batch))
                val_losses.append(val_result["loss"])
            val_loss = np.mean(val_losses)
            print(f"Validation Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("Early stopping triggered")
                    break

        # Evaluate on test data
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, test_adj_matrices, y_test))
        test_dataset = test_dataset.batch(batch_size)

        predictions = []
        test_loss = []
        for x_batch, adj_batch, y_batch in test_dataset:
            result = model.test_step((x_batch, adj_batch, y_batch))
            test_loss.append(result["loss"])
            predictions.append(result["y_pred"].numpy())

        avg_test_loss = np.mean(test_loss)
        print(f"Average test loss for window {window_index + 1}: {avg_test_loss:.4f}")

        # Save predictions and true values
        predictions = np.concatenate(predictions, axis=0)
        np.save(os.path.join(RESULTS_DIR, f"predictions_window_{window_index}.npy"), predictions)
        np.save(os.path.join(RESULTS_DIR, f"y_test_window_{window_index}.npy"), y_test)

        # Calculate and save rankings
        rankings = np.argsort(-predictions, axis=1)
        np.save(os.path.join(RESULTS_DIR, f"rankings_window_{window_index}.npy"), rankings)

        # Visualize results (optional, you might want to adjust this for multiple windows)
        plot_values_time(predictions, asset_names=asset_names, title_=f'Predictions (Window {window_index + 1})')
        plot_values_time(y_test, asset_names=asset_names, title_=f'True Returns (Window {window_index + 1})')


if __name__ == "__main__":
    main()
