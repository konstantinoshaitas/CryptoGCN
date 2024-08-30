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

# Load Denoised Correlation Matrices
correlation_data = pd.read_csv(AGGREGATED_DATA_PATH, index_col=0)
correlation_matrix = CorrelationMatrix(correlation_data, window_size=SEQUENCE_LENGTH)
train_denoised, valid_denoised, test_denoised = correlation_matrix.run()


def load_and_process_aggregated_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    # Print Columns
    print("Column names in the dataset:", df.columns)
    # Ensure data is sorted by UTC time
    df.sort_values(by='time', inplace=True)

    # Calculate t+1 returns for each asset
    for col in df.columns:
        if col != 'time':  # Skip the time column
            df[f'{col}_t+1_return'] = df[col].pct_change().shift(-1)

    # Drop the original columns (keep only the return columns and time)
    return df[['time'] + [col for col in df.columns if '_t+1_return' in col]]


# Function to prepare the dataset
def prepare_dataset():
    # Load the aggregated asset data
    df = load_and_process_aggregated_data(AGGREGATED_DATA_PATH)

    # Extract asset names (excluding the time column)
    asset_names = df.columns[1:]  # Assuming first column is 'time' and the rest are assets

    # Split the data into training, validation, and testing sets
    train_index = int(len(df) * 0.7)  # 70% for training
    valid_index = int(len(df) * 0.80)  # 10% for validation, 20% for testing

    train_df = df.iloc[:train_index]
    valid_df = df.iloc[train_index:valid_index]
    test_df = df.iloc[valid_index:]

    # Prepare the LSTM input data and targets
    def create_sequences_and_targets(data):
        sequences = []
        targets = []
        for i in range(len(data) - SEQUENCE_LENGTH):
            seq = data.iloc[i:i + SEQUENCE_LENGTH, 1:].values  # Exclude the time column
            target = data.iloc[i + SEQUENCE_LENGTH, 1:].values  # Target is the t+1 return
            sequences.append(seq)
            targets.append(target)
        return np.array(sequences), np.array(targets)

    x_train, y_train = create_sequences_and_targets(train_df)
    x_valid, y_valid = create_sequences_and_targets(valid_df)
    x_test, y_test = create_sequences_and_targets(test_df)

    # Correct the slicing of adjacency matrices
    train_adj_matrices = np.array(train_denoised[:len(x_train)])  # Matches x_train
    valid_adj_matrices = np.array(valid_denoised[:len(x_valid)])  # Matches x_valid
    test_adj_matrices = np.array(test_denoised[:len(x_test)])  # Matches x_test

    # Check shapes
    print(f"x_train shape: {x_train.shape}")
    print(f"x_valid shape: {x_valid.shape}")
    print(f"x_test shape: {x_test.shape}")

    print(f"y_train shape: {y_train.shape}")
    print(f"y_valid shape: {y_valid.shape}")
    print(f"y_test shape: {y_test.shape}")

    print(f"train_adj_matrices shape: {train_adj_matrices.shape}")
    print(f"valid_adj_matrices shape: {valid_adj_matrices.shape}")
    print(f"test_adj_matrices shape: {test_adj_matrices.shape}")

    return x_train, y_train, train_adj_matrices, x_valid, y_valid, valid_adj_matrices, x_test, y_test, test_adj_matrices, asset_names


def main():
    # Load and prepare data
    batch_size = 64
    x_train, y_train, train_adj_matrices, x_valid, y_valid, valid_adj_matrices, x_test, y_test, test_adj_matrices, asset_names = prepare_dataset()

    # Convert data to float32 and ensure correct shape
    x_train = x_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_valid = y_valid.astype(np.float32)
    y_test = y_test.astype(np.float32)

    x_test = x_test[:-1]
    y_test = y_test[:-1]
    test_adj_matrices = test_adj_matrices[:-1]

    print(f"x_valid shape: {x_valid.shape}")
    print(f"valid_adj_matrices shape: {valid_adj_matrices.shape}")
    print(f"y_valid shape: {y_valid.shape}")

    # Define the model
    model = EndToEndCryptoModel(sequence_length=SEQUENCE_LENGTH, lstm_units=5, num_assets=x_train.shape[2], alpha=1)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.000005))

    # Train and validation data preparation
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, train_adj_matrices, y_train))
    train_dataset = train_dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, valid_adj_matrices, y_valid))
    valid_dataset = valid_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    # Calculate steps per epoch based on the available train data
    steps_per_epoch = len(y_train) // batch_size

    epochs = 6
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

        # Early Stopping (optional)
        if avg_valid_loss < best_val_loss:
            best_val_loss = avg_valid_loss
            print("Validation loss improved, saving the model.")
            # Save the model weights if validation loss improved
            model.save_weights("best_model.weights.h5")
        else:
            print("No improvement in validation loss.")

    # Evaluation with Predictions on Test Set
    print("\nEvaluating the model on the test set...")
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, test_adj_matrices, y_test))
    test_dataset = test_dataset.batch(batch_size)

    predictions = []
    test_loss = []
    for x_batch, adj_batch, y_batch in test_dataset:
        result = model.test_step((x_batch, adj_batch, y_batch))
        test_loss.append(result["loss"])
        predictions.append(result["y_pred"].numpy())  # Save the predictions

    avg_test_loss = np.mean(test_loss)
    print(f"Average test loss: {avg_test_loss:.4f}")

    # Convert predictions to a numpy array
    predictions = np.concatenate(predictions, axis=0)
    np.save(os.path.join(RESULTS_DIR, "predictions.npy"), predictions)
    np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)

    # Calculate rankings for each time step
    rankings = np.argsort(-predictions, axis=1)  # Rank in descending order, highest return is rank 0
    np.save(os.path.join(RESULTS_DIR, "rankings.npy"), rankings)

    plot_values_time(predictions, asset_names=asset_names, title_='Predictions')
    plot_values_time(y_test, asset_names=asset_names, title_='True Returns')

    # Save model
    model.save("crypto_lstm_gcn_model.keras")


if __name__ == "__main__":
    main()
