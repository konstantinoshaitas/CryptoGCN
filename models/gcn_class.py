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
    def __init__(self, denoised_matrices):
        self.denoised_matrices = denoised_matrices

    def build_graph(self, denoised_matrix):
        return denoised_matrix

    def graph_convolution(self, num_assets, hidden_state_dim, adjacency_matrix):
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

    def apply_gcn(self, sequential_embeddings):
        gcn_outputs = []
        num_time_steps, num_assets, hidden_state_dim = sequential_embeddings.shape

        for t, denoised_matrix in enumerate(self.denoised_matrices):
            adjacency_matrix = self.build_graph(denoised_matrix)
            gcn_model = self.graph_convolution(num_assets, hidden_state_dim, adjacency_matrix)
            gcn_output = gcn_model.predict([sequential_embeddings[t:t + 1], adjacency_matrix[np.newaxis, ...]])
            gcn_outputs.append(gcn_output)

        return gcn_outputs