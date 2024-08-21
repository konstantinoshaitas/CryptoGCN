import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Flatten, Layer, BatchNormalization
import numpy as np

'''
GCN CLASS
'''


class GraphConvolution(Layer):
    def __init__(self, units, activation=None, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.units = units
        self.activation = tf.keras.layers.LeakyReLU(negative_slope=0.01)
        self.kernel_initializer = tf.keras.initializers.he_normal()

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


'''
End-to-End Model combining LSTM and GCN
'''


class EndToEndCryptoModel(Model):
    def __init__(self, sequence_length, num_assets, lstm_units=15, gcn_units1=16, gcn_units2=8, alpha=0.5):
        super(EndToEndCryptoModel, self).__init__()
        self.lstm = LSTM(units=lstm_units, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(0.008))
        self.batch_norm_lstm = BatchNormalization()
        self.dropout = Dropout(0.1)
        self.gcn1 = GraphConvolution(gcn_units1, activation='leaky_relu')
        self.gcn2 = GraphConvolution(gcn_units2, activation='leaky_relu')
        self.batch_norm_gcn1 = BatchNormalization()
        self.batch_norm_gcn2 = BatchNormalization()
        self.flatten = Flatten()
        self.dense1 = Dense(3, activation='relu')
        self.dense2 = Dense(num_assets, activation='linear')
        self.alpha = alpha
        self.mse_loss = tf.keras.losses.MeanSquaredError()

        self.sequence_length = sequence_length

    def call(self, inputs):
        x, a = inputs  # X: shape (batch_size, sequence_length=21, num_assets=15), A: adjacency matrix

        # Transpose X to shape [batch_size, sequence_length, num_assets]
        x = tf.transpose(x, perm=[0, 1, 2])  # Keeping the shape [batch_size, sequence_length, num_assets]

        # Process the entire sequence for all assets through the LSTM
        lstm_outputs = self.lstm(x)  # Shape: [batch_size, sequence_length, lstm_units]
        lstm_outputs = self.batch_norm_lstm(lstm_outputs)

        # Apply Dropout
        lstm_outputs = self.dropout(lstm_outputs)  # Shape remains [batch_size, sequence_length, lstm_units]

        # print(f'lstm_output_shape: {lstm_outputs.numpy().shape}')
        # tf.print("LSTM Outputs:", lstm_outputs, summarize=1)

        # Now, iterate over each time step to pass the output through the GCN
        gcn_outputs = []
        for t in range(lstm_outputs.shape[1]):  # Iterate over sequence_length
            lstm_output_at_t = lstm_outputs[:, t, :]  # Extract LSTM output for all assets at time step t
            lstm_output_at_t = tf.expand_dims(lstm_output_at_t, axis=1)  # Shape: [batch_size, 1, lstm_units]

            # Reshape the LSTM output to match the expected input shape for GCN
            lstm_output_at_t = tf.tile(lstm_output_at_t,
                                       [1, x.shape[2], 1])  # Shape: [batch_size, num_assets, lstm_units]

            # Ensure the GCN is receiving varied inputs
            # tf.print("GCN Input at time step", t, ":", lstm_output_at_t, summarize=1)

            # Feed the LSTM output at this time step into the GCN
            gcn_output = self.gcn1([lstm_output_at_t, a])  # Shape: [batch_size, num_assets, gcn_units1]
            gcn_output = self.batch_norm_gcn1(gcn_output)
            # tf.print("GCN Outputs after gcn1 at time step", t, ":", gcn_output, summarize=1)

            gcn_output = self.gcn2([gcn_output, a])  # Shape: [batch_size, num_assets, gcn_units2]
            gcn_output = self.batch_norm_gcn2(gcn_output)
            # tf.print("GCN Output after gcn2 at time step", t, ":", gcn_output, summarize=1)

            gcn_outputs.append(gcn_output)

        # Stack the GCN outputs for all time steps
        gcn_outputs = tf.stack(gcn_outputs, axis=1)  # Shape: [batch_size, sequence_length, num_assets, gcn_units2]
        # print(f'GCN Output Shape: {gcn_outputs.shape}')

        # Flatten and pass through dense layers
        x = self.flatten(gcn_outputs)  # Shape: [batch_size, sequence_length * num_assets * gcn_units2]
        x = self.dense1(x)  # Shape: [batch_size, 64]

        final_output = self.dense2(x)
        # tf.print("Final Output:", final_output, summarize=1)

        return final_output  # Shape: [batch_size, num_assets]

    def train_step(self, data):
        x, a, y_true = data
        # print(f"X shape: {X.shape}, A shape: {A.shape}, y_true shape: {y_true.shape}")

        with tf.GradientTape() as tape:
            y_pred = self([x, a], training=True)
            loss = self.combined_loss(y_true, y_pred)

        gradients = tape.gradient(loss, self.trainable_variables)
        clipped_gradients = [tf.clip_by_value(grad, -1.0, 1.0) for grad in gradients]

        # gradients_norm = [tf.norm(g) for g in gradients]
        # tf.print("Gradient norms:", gradients_norm)
        clipped_gradients_norm = [tf.norm(g) for g in clipped_gradients]
        # tf.print("Clipped Gradient norms:", clipped_gradients_norm)

        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return {"loss": loss}

    def test_step(self, data):
        x, a, y_true = data
        y_pred = self([x, a], training=False)
        # print(f'y_pred:\n{y_pred} \n vs \n y_true:\n{y_true}')
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
