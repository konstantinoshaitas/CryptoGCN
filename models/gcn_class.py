import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
import numpy as np


class GraphConvolution(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer='glorot_uniform',
                                      name='kernel')
        self.bias = self.add_weight(shape=(self.units,),
                                    initializer='zeros',
                                    name='bias')
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs):
        features, adjacency = inputs
        support = tf.matmul(features, self.kernel)
        output = tf.matmul(adjacency, support)
        output = output + self.bias
        return self.activation(output) if self.activation is not None else output


class CryptoGCN:
    def __init__(self, train_denoised_matrices, test_denoised_matrices):
        self.train_denoised_matrices = train_denoised_matrices
        self.test_denoised_matrices = test_denoised_matrices
        self.model = None

    def build_graph(self, denoised_matrix):
        return denoised_matrix

    def graph_convolution(self, num_assets, hidden_state_dim):
        X_input = Input(shape=(num_assets, hidden_state_dim))
        A_input = Input(shape=(num_assets, num_assets))

        # First GCN layer
        gcn_output = GraphConvolution(64, activation='relu')([X_input, A_input])

        # Second GCN layer
        gcn_output = GraphConvolution(32, activation='relu')([gcn_output, A_input])

        # Flatten the output
        gcn_output = tf.keras.layers.Flatten()(gcn_output)

        # Final dense layer for ranking scores
        output = Dense(num_assets, activation='linear')(gcn_output)

        model = Model(inputs=[X_input, A_input], outputs=output)
        model.compile(optimizer='adam', loss='mse')
        return model

    def apply_gcn(self, train_hidden_states, test_hidden_states, batch_size=32):
        num_time_steps, num_assets, hidden_state_dim = train_hidden_states.shape

        # Build and compile the model only once
        self.model = self.graph_convolution(num_assets, hidden_state_dim)

        train_outputs = self.process_data(train_hidden_states, self.train_denoised_matrices, batch_size)
        test_outputs = self.process_data(test_hidden_states, self.test_denoised_matrices, batch_size)
        return train_outputs, test_outputs

    def process_data(self, hidden_states, denoised_matrices, batch_size):
        num_time_steps = len(hidden_states)
        gcn_outputs = []

        for i in range(0, num_time_steps, batch_size):
            batch_end = min(i + batch_size, num_time_steps)
            batch_hidden_states = hidden_states[i:batch_end]
            batch_matrices = denoised_matrices[i:batch_end]

            # Prepare inputs for the batch
            X_batch = np.array(batch_hidden_states)
            A_batch = np.array([self.build_graph(matrix) for matrix in batch_matrices])

            # Predict for the batch
            batch_outputs = self.model.predict([X_batch, A_batch], verbose=0)
            gcn_outputs.extend(batch_outputs)

        return np.array(gcn_outputs)