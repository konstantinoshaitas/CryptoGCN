import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten, Layer
import numpy as np

# LSTM Class
class CryptoLSTM(Model):
    def __init__(self, sequence_length, lstm_units=40):
        super(CryptoLSTM, self).__init__()
        self.lstm = LSTM(units=lstm_units, return_sequences=False, return_state=True, kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dropout = Dropout(0.2)

    def call(self, inputs):
        x, state_h, state_c = self.lstm(inputs)
        x = self.dropout(state_h)  # Use the last hidden state
        return x  # Return the hidden state

# GCN Class
class GraphConvolution(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = input_shape[0][-1]
        self.kernel = self.add_weight(shape=(input_dim, self.units), initializer='glorot_uniform', name='kernel')
        self.bias = self.add_weight(shape=(self.units,), initializer='zeros', name='bias')
        super(GraphConvolution, self).build(input_shape)

    def call(self, inputs):
        features, adjacency = inputs
        support = tf.matmul(features, self.kernel)
        output = tf.matmul(adjacency, support)
        output = output + self.bias
        return self.activation(output) if self.activation is not None else output


# End-to-End Model combining LSTM and GCN
class EndToEndCryptoModel(Model):
    def __init__(self, sequence_length, num_assets, lstm_units=40, gcn_units1=64, gcn_units2=32, alpha=0.85):
        super(EndToEndCryptoModel, self).__init__()
        self.lstm = CryptoLSTM(sequence_length, lstm_units)
        self.gcn1 = GraphConvolution(gcn_units1, activation='relu')
        self.gcn2 = GraphConvolution(gcn_units2, activation='relu')
        self.flatten = Flatten()
        self.dense1 = Dense(64, activation='relu')
        self.dense2 = Dense(num_assets, activation='linear')
        self.alpha = alpha
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def call(self, inputs):
        X, A = inputs  # X is the input data, A is the adjacency matrix

        # Process each asset separately with the LSTM
        lstm_outputs = []
        for i in range(X.shape[2]):  # X.shape[2] is the num_assets
            lstm_input = X[:, :, i]  # Extract the time series for one asset
            lstm_input = tf.expand_dims(lstm_input, -1)  # Add an extra dimension for num_features

            # if i == 0:
            #     print(f"LSTM Input Sequence for Asset {i}:\n", lstm_input.numpy())

            lstm_output = self.lstm(lstm_input)
            lstm_outputs.append(lstm_output)

        # Combine the LSTM outputs into a single matrix
        lstm_outputs = tf.stack(lstm_outputs, axis=1)  # Shape: [batch_size, num_assets, lstm_units]

        # Feed the combined LSTM outputs to the GCN using the adjacency matrix
        x = self.gcn1([lstm_outputs, A])
        x = self.gcn2([x, A])
        x = self.flatten(x)
        x = self.dense1(x)
        return self.dense2(x)



    def train_step(self, data):
        X, A, y_true = data

        with tf.GradientTape() as tape:
            y_pred = self([X, A], training=True)
            loss = self.combined_loss(y_true, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

    def test_step(self, data):
        X, A, y_true = data
        y_pred = self([X, A], training=False)
        loss = self.combined_loss(y_true, y_pred)
        return {"loss": loss, "y_pred": y_pred}

    def pairwise_ranking_loss(self, y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1, y_true.shape[-1]]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1, y_pred.shape[-1]]), tf.float32)

        diff_true = y_true[:, :, None] - y_true[:, None, :]
        diff_pred = y_pred[:, :, None] - y_pred[:, None, :]

        loss = tf.maximum(tf.constant(0, dtype=tf.float32),
                          -tf.sign(diff_true) * diff_pred + tf.constant(0.1, dtype=tf.float32))
        return tf.reduce_mean(loss)

    def combined_loss(self, y_true, y_pred):
        mse_loss = self.mse_loss(y_true, y_pred)
        ranking_loss = self.pairwise_ranking_loss(y_true, y_pred)
        return (1 - self.alpha) * mse_loss + self.alpha * ranking_loss
