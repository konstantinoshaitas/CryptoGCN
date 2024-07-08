import numpy as np
import tensorflow as tf
from spektral.data import Dataset, Graph
from spektral.data.loaders import SingleLoader
from spektral.layers import GCNConv

# Assuming price_data is a NumPy array of shape (num_cryptos, time_steps)
# Calculate standard returns
returns = np.diff(price_data) / price_data[:, :-1]


# Calculate volatilities using two methods
def calculate_volatility(returns, window_size=48):
    vol_std = np.zeros_like(returns)
    vol_ma = np.zeros_like(returns)
    for i in range(window_size, returns.shape[1]):
        window = returns[:, i - window_size:i]
        vol_std[:, i] = np.std(window, axis=1)
        vol_ma[:, i] = np.sqrt(np.mean(window ** 2, axis=1))
    return vol_std, vol_ma


vol_std, vol_ma = calculate_volatility(returns)


# Compute correlation matrices using Pearson correlation coefficient
def compute_correlation_matrices(returns, vol_std, vol_ma, window_size=48):
    correlation_matrices = []
    for i in range(window_size, returns.shape[1]):
        z_std = (returns[:, i - window_size:i] - vol_std[:, i - window_size:i].mean(axis=1, keepdims=True)) / vol_std[:,
                                                                                                              i - window_size:i].std(
            axis=1, keepdims=True)
        z_ma = (returns[:, i - window_size:i] - vol_ma[:, i - window_size:i].mean(axis=1, keepdims=True)) / vol_ma[:,
                                                                                                            i - window_size:i].std(
            axis=1, keepdims=True)
        corr_matrix_std = np.corrcoef(z_std)
        corr_matrix_ma = np.corrcoef(z_ma)
        correlation_matrices.append((corr_matrix_std, corr_matrix_ma))
    return correlation_matrices


correlation_matrices = compute_correlation_matrices(returns, vol_std, vol_ma)

# Extract standard correlation matrices for further processing
correlation_matrices = [cm[0] for cm in correlation_matrices]  # Using std method for example

# Create adjacency matrices based on correlation matrices
threshold = 0.5
adj_matrices = []

for correlation_matrix in correlation_matrices:
    edge_index = []
    edge_weight = []
    num_cryptos = correlation_matrix.shape[0]

    for i in range(num_cryptos):
        for j in range(i + 1, num_cryptos):
            if correlation_matrix[i, j] > threshold:
                edge_index.append([i, j])
                edge_index.append([j, i])
                edge_weight.append(correlation_matrix[i, j])
                edge_weight.append(correlation_matrix[i, j])

    edge_index = np.array(edge_index)
    edge_weight = np.array(edge_weight)

    # Create adjacency matrix
    adj = np.zeros((num_cryptos, num_cryptos))
    for i, j, w in zip(edge_index[:, 0], edge_index[:, 1], edge_weight):
        adj[int(i), int(j)] = w

    adj_matrices.append(adj)

adj_matrices = np.array(adj_matrices)  # Shape: (num_windows, num_cryptos, num_cryptos)


class TemporalCryptoDataset(Dataset):
    def __init__(self, node_features, adj_matrices, **kwargs):
        self.node_features = node_features
        self.adj_matrices = adj_matrices
        super().__init__(**kwargs)

    def read(self):
        graphs = []
        for i in range(self.adj_matrices.shape[0]):
            x = self.node_features
            a = tf.sparse.from_dense(self.adj_matrices[i])
            graphs.append(Graph(x=x, a=a))
        return graphs


dataset = TemporalCryptoDataset(node_features, adj_matrices)
loader = SingleLoader(dataset)

# Define the Temporal Graph Convolution (TGC) Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate


class TemporalGraphConv(Model):
    def __init__(self, gcn_units, dense_units):
        super().__init__()
        self.gcn1 = GCNConv(gcn_units, activation='relu')
        self.gcn2 = GCNConv(gcn_units, activation='relu')
        self.dense = Dense(dense_units, activation='relu')

    def call(self, inputs):
        x, a = inputs
        x = self.gcn1([x, a])
        x = self.gcn2([x, a])
        return x


class TGCModel(Model):
    def __init__(self, gcn_units, dense_units):
        super().__init__()
        self.temporal_graph_conv = TemporalGraphConv(gcn_units, dense_units)
        self.fc = Dense(1)  # Output ranking score

    def call(self, inputs):
        x, a = inputs
        gcn_output = self.temporal_graph_conv([x, a])
        concatenated = Concatenate()([x, gcn_output])  # Combine LSTM embeddings with GCN output
        ranking_score = self.fc(concatenated)
        return ranking_score


# Instantiate the model
gcn_units = 16
dense_units = 16
tgc_model = TGCModel(gcn_units, dense_units)

# Compile and Train the Model
tgc_model.compile(optimizer='adam', loss='mse')

# Assume target_ranking_scores is a TensorFlow tensor with shape (num_windows, num_cryptos, 1)
# For demonstration purposes, let's create dummy target ranking scores
target_ranking_scores = tf.random.normal((len(adj_matrices), num_cryptos, 1))

# Training loop
epochs = 200
for epoch in range(epochs):
    loss = 0
    for i in range(len(adj_matrices)):
        data = (node_features, adj_matrices[i])
        loss += tgc_model.train_on_batch(data, target_ranking_scores[i])
    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss / len(adj_matrices)}')

# Evaluate the model
predictions = []
for i in range(len(adj_matrices)):
    data = (node_features, adj_matrices[i])
    predictions.append(tgc_model.predict(data))
predictions = np.array(predictions)
