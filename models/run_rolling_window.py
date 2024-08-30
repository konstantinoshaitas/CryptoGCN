import os
import pandas as pd
import numpy as np
import tensorflow as tf
from Crypto_LSTM_GCN import EndToEndCryptoModel
from pearson_correlation import CorrelationMatrix
from visualisations import plot_values_time
import random

SEED = 100
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Load Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed_sequential_data')
AGGREGATED_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'correlation_data', 'aggregated_asset_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results', 'rolling_window_results')


# Parameters for Rolling Window
WINDOW_SIZE = 3392  # Adjustable: number of time steps in each rolling window
STEP_SIZE = 100  # Adjustable: number of time steps to roll forward each iteration
SEQUENCE_LENGTH = 21


def load_and_process_aggregated_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    df.sort_values(by='time', inplace=True)
    for col in df.columns:
        if col != 'time':
            df[f'{col}_t+1_return'] = df[col].pct_change().shift(-1)
    return df[['time'] + [col for col in df.columns if '_t+1_return' in col]]


def create_sequences_and_targets(data, sequence_length):
    sequences = []
    targets = []
    for i in range(len(data) - sequence_length):
        seq = data.iloc[i:i + sequence_length, 1:].values
        target = data.iloc[i + sequence_length, 1:].values
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)


def prepare_rolling_window_dataset(df, window_size, step_size):
    datasets = []
    num_samples = len(df)

    for start in range(0, num_samples, step_size):
        end = start + window_size

        # Stop if there's not enough data left for a full window
        if end + SEQUENCE_LENGTH > num_samples:
            print(f"Stopping early at window starting at index {start}, not enough data for full window.")
            break

        # Prepare the windowed dataset
        window_df = df.iloc[start:end + SEQUENCE_LENGTH]
        correlation_matrix = CorrelationMatrix(window_df.iloc[:, 1:], window_size=SEQUENCE_LENGTH)
        denoised_matrices = correlation_matrix.run()

        # Create train, valid, test splits and prepare sequences and targets
        train_index = int(len(window_df) * 0.7)
        valid_index = int(len(window_df) * 0.80)
        train_df = window_df.iloc[:train_index]
        valid_df = window_df.iloc[train_index:valid_index]
        test_df = window_df.iloc[valid_index:]

        x_train, y_train = create_sequences_and_targets(train_df, SEQUENCE_LENGTH)
        x_valid, y_valid = create_sequences_and_targets(valid_df, SEQUENCE_LENGTH)
        x_test, y_test = create_sequences_and_targets(test_df, SEQUENCE_LENGTH)

        train_adj_matrices = np.array(denoised_matrices[0][:len(x_train)])
        valid_adj_matrices = np.array(denoised_matrices[1][:len(x_valid)])
        test_adj_matrices = np.array(denoised_matrices[2][-len(x_test):])

        datasets.append((x_train, y_train, train_adj_matrices, x_valid, y_valid, valid_adj_matrices, x_test, y_test, test_adj_matrices))

    return datasets


def main():
    # Load and prepare data
    df = load_and_process_aggregated_data(AGGREGATED_DATA_PATH)
    rolling_datasets = prepare_rolling_window_dataset(df, WINDOW_SIZE, STEP_SIZE)

    for roll_index, (x_train, y_train, train_adj_matrices, x_valid, y_valid, valid_adj_matrices, x_test, y_test, test_adj_matrices) in enumerate(
            rolling_datasets):
        print(f"Running roll {roll_index + 1}/{len(rolling_datasets)}")

        x_train = x_train.astype(np.float32)
        x_valid = x_valid.astype(np.float32)
        x_test = x_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_valid = y_valid.astype(np.float32)
        y_test = y_test.astype(np.float32)

        # Define the model
        model = EndToEndCryptoModel(sequence_length=SEQUENCE_LENGTH, lstm_units=5, num_assets=x_train.shape[2], alpha=1)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005))

        # Train and validation data preparation
        batch_size = 64
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, train_adj_matrices, y_train))
        train_dataset = train_dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

        valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, valid_adj_matrices, y_valid))
        valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = len(y_train) // batch_size
        epochs = 5
        best_val_loss = float('inf')

        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            # Training Loop
            for step, (x_batch, adj_batch, y_batch) in enumerate(train_dataset.take(steps_per_epoch)):
                loss = model.train_step((x_batch, adj_batch, y_batch))
                if step % 10 == 0:
                    print(f"Step {step}: Loss = {loss['loss']:.4f}")

            # Validation Loop
            valid_loss = []
            for x_batch, adj_batch, y_batch in valid_dataset:
                result = model.test_step((x_batch, adj_batch, y_batch))
                valid_loss.append(result["loss"])

            avg_valid_loss = np.mean(valid_loss)
            print(f"Validation Loss after Epoch {epoch + 1}: {avg_valid_loss:.4f}")

            if avg_valid_loss < best_val_loss:
                best_val_loss = avg_valid_loss
                print("Validation loss improved, saving the model.")
                model.save_weights(f"best_model_roll_{roll_index}.weights.h5")
            else:
                print("No improvement in validation loss.")

        print("\nEvaluating the model on the test set...")
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test, test_adj_matrices, y_test))
        test_dataset = test_dataset.batch(batch_size)

        predictions = []
        test_loss = []
        for x_batch, adj_batch, y_batch in test_dataset:
            result = model.test_step((x_batch, adj_batch, y_batch))
            test_loss.append(result["loss"])
            predictions.append(result["y_pred"].numpy())

        avg_test_loss = np.mean(test_loss)
        print(f"Average test loss for roll {roll_index + 1}: {avg_test_loss:.4f}")

        predictions = np.concatenate(predictions, axis=0)
        np.save(os.path.join(RESULTS_DIR, f"predictions_roll_{roll_index}.npy"), predictions)
        np.save(os.path.join(RESULTS_DIR, f"y_test_roll_{roll_index}.npy"), y_test)

        rankings = np.argsort(-predictions, axis=1)
        np.save(os.path.join(RESULTS_DIR, f"rankings_roll_{roll_index}.npy"), rankings)

        # plot_values_time(predictions, asset_names=df.columns[1:], title_=f'Predictions Roll {roll_index + 1}')
        # plot_values_time(y_test, asset_names=df.columns[1:], title_=f'True Returns Roll {roll_index + 1}')


if __name__ == "__main__":
    main()
