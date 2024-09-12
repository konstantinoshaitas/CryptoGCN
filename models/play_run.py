import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import plot_model
from pearson_correlation import CorrelationMatrix
import pandas as pd
import random
from Crypto_LSTM_GCN import EndToEndCryptoModel
import visualkeras

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Load the saved model
with tf.keras.utils.custom_object_scope({'EndToEndCryptoModel': EndToEndCryptoModel}):
    model = load_model("crypto_lstm_gcn_model.keras")
print('LSTM-GCN Model Loaded...')

model.summary()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AGGREGATED_DATA_PATH = os.path.join(BASE_DIR, '..', 'play_data', 'aggregated_asset_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, '..', 'results', 'play_results')

# Model Parameters
SEQUENCE_LENGTH = 21

# Load Denoised Correlation Matrices
correlation_data = pd.read_csv(AGGREGATED_DATA_PATH, index_col=0)
correlation_matrix = CorrelationMatrix(correlation_data, window_size=SEQUENCE_LENGTH)
train_denoised, valid_denoised, test_denoised = correlation_matrix.run()
denoised_matrices = np.concatenate((train_denoised, valid_denoised, test_denoised), axis=0)


# Load and preprocess the new data
def load_and_process_new_data(file_path, sequence_length=21):
    df = pd.read_csv(file_path, parse_dates=['time'])
    df.sort_values(by='time', inplace=True)

    for col in df.columns:
        if col != 'time':  # Skip the time column
            df[f'{col}_t+1_return'] = df[col].pct_change().shift(-1)

    processed_df = df[['time'] + [col for col in df.columns if '_t+1_return' in col]]

    sequences = []
    y_test = []
    for i in range(len(processed_df) - sequence_length):
        seq = processed_df.iloc[i:i + sequence_length, 1:].values  # Exclude the time column
        sequences.append(seq)
        y_test.append(processed_df.iloc[i + sequence_length, 1:].values)  # True returns for the prediction

    sequences = np.array(sequences)
    y_test = np.array(y_test)
    return sequences, y_test


# Prepare the data (this should match the data preprocessing during training)
new_sequences, y_test = load_and_process_new_data(AGGREGATED_DATA_PATH)

# Adjust the length of new sequences to match the correlation matrices
new_sequences = new_sequences[:len(denoised_matrices)]
y_test = y_test[:len(denoised_matrices)]

# Make predictions using the loaded model
predictions = model.predict([new_sequences, denoised_matrices])

# Rank the predictions
rankings = np.argsort(-predictions, axis=1)

# Save the predictions, rankings, and y_test as NumPy files
np.save(os.path.join(RESULTS_DIR, "rankings.npy"), rankings)
np.save(os.path.join(RESULTS_DIR, "predictions.npy"), predictions)
np.save(os.path.join(RESULTS_DIR, "y_test.npy"), y_test)

# Output the predictions
print("Predictions:", predictions.shape)
print("y_test:", y_test.shape)

print(y_test)

# plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True)
# model.summary()
# visualkeras.layered_view(model, to_file='model_architecture_visualkeras.png').show()
