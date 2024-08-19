import os
import pandas as pd
import numpy as np
import tensorflow as tf
from end_end_class import EndToEndCryptoModel
from pearson_correlation import CorrelationMatrix

# Load Directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '..', 'data', 'processed_sequential_data')
AGGREGATED_DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'correlation_data', 'aggregated_asset_data.csv')

# Model Parameters
SEQUENCE_LENGTH = 21
TRAIN_TEST_SPLIT = 0.75

# Load Denoised Correlation Matrices
correlation_data = pd.read_csv(AGGREGATED_DATA_PATH, index_col=0)
correlation_matrix = CorrelationMatrix(correlation_data, window_size=SEQUENCE_LENGTH)
train_denoised, test_denoised = correlation_matrix.run()


def load_and_process_aggregated_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['time'])
    # Print Columns
    print("Column names in the dataset:", df.columns)
    # Ensure data is sorted by UTC time
    df.sort_values(by='time', inplace=True)

    # Calculate t+1 returns for each asset
    for col in df.columns:
        if col != 'time':  # Skip the time column
            df[f'{col}_t+1_return'] = df[col].pct_change().shift(-1).fillna(0)

    # Drop the original columns (keep only the return columns and time)
    return df[['time'] + [col for col in df.columns if '_t+1_return' in col]]


# Function to prepare the dataset
def prepare_dataset():
    # Load the aggregated asset data
    df = load_and_process_aggregated_data(AGGREGATED_DATA_PATH)

    # Split the data into training and testing sets
    split_index = int(len(df) * TRAIN_TEST_SPLIT)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

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
    x_test, y_test = create_sequences_and_targets(test_df)

    # Use the correlation matrices as adjacency matrices
    train_adj_matrices = np.array(train_denoised[:len(x_train)])  # Trim to match x_train length
    test_adj_matrices = np.array(test_denoised[:len(x_test)])  # Trim to match x_test length

    return x_train, y_train, train_adj_matrices, x_test, y_test, test_adj_matrices


def main():
    # Load and prepare data
    x_train, y_train, train_adj_matrices, x_test, y_test, test_adj_matrices = prepare_dataset()

    # Convert data to float32 and ensure correct shape
    x_train = x_train.astype(np.float32)
    x_test = x_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    # Define the model
    model = EndToEndCryptoModel(sequence_length=SEQUENCE_LENGTH, num_assets=x_train.shape[2])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

    # Train data preparation
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, train_adj_matrices, y_train))
    train_dataset = train_dataset.batch(64).repeat().prefetch(tf.data.AUTOTUNE)

    # Calculate steps per epoch based on the available train data
    steps_per_epoch = len(y_train) // 64

    epochs = 1
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for step, (x_batch, adj_batch, y_batch) in enumerate(train_dataset.take(steps_per_epoch)):
            loss = model.train_step((x_batch, adj_batch, y_batch))
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss['loss']:.4f}")

    # Evaluation
    print("\nEvaluating the model on the test set...")
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, test_adj_matrices, y_test))
    test_dataset = test_dataset.batch(64)

    test_loss = []
    for x_batch, adj_batch, y_batch in test_dataset:
        loss = model.test_step((x_batch, adj_batch, y_batch))
        test_loss.append(loss['loss'])

    avg_test_loss = np.mean(test_loss)
    print(f"Average test loss: {avg_test_loss:.4f}")


if __name__ == "__main__":
    main()
