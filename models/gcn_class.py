import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer
from tensorflow.keras.optimizers import Adam
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


class CryptoGCN(Model):
    def __init__(self, num_assets, hidden_state_dim):
        super(CryptoGCN, self).__init__()
        self.gcn1 = GraphConvolution(64, activation='relu')
        self.gcn2 = GraphConvolution(32, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense = Dense(num_assets, activation='linear')
        self.alpha = 0.5  # Hyperparameter to balance pointwise and pairwise loss
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    def call(self, inputs):
        X, A = inputs
        x = self.gcn1([X, A])
        x = self.gcn2([x, A])
        x = self.flatten(x)
        return self.dense(x)

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
        return {"loss": loss}

    def combined_loss(self, y_true, y_pred):
        pointwise_loss = self.mse_loss(y_true, y_pred)
        pairwise_loss = self.pairwise_ranking_loss(y_true, y_pred)
        return pointwise_loss + self.alpha * pairwise_loss

    def pairwise_ranking_loss(self, y_true, y_pred):
        # Ensure y_true and y_pred are 2D and float32
        y_true = tf.cast(tf.reshape(y_true, [-1, y_true.shape[-1]]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1, y_pred.shape[-1]]), tf.float32)

        diff_true = y_true[:, :, None] - y_true[:, None, :]
        diff_pred = y_pred[:, :, None] - y_pred[:, None, :]

        loss = tf.maximum(tf.constant(0, dtype=tf.float32),
                          -tf.sign(diff_true) * diff_pred + tf.constant(0.1, dtype=tf.float32))
        return tf.reduce_mean(loss)


def apply_gcn(train_hidden_states, test_hidden_states, train_denoised_matrices, test_denoised_matrices, train_y,
              test_y):
    print("Shape of train_hidden_states:", train_hidden_states.shape)
    print("Shape of train_denoised_matrices:", train_denoised_matrices.shape)
    print("Shape of train_y:", train_y.shape)
    print("Shape of test_hidden_states:", test_hidden_states.shape)
    print("Shape of test_denoised_matrices:", test_denoised_matrices.shape)
    print("Shape of test_y:", test_y.shape)

    num_time_steps, num_assets, hidden_state_dim = train_hidden_states.shape

    model = CryptoGCN(num_assets, hidden_state_dim)
    model.compile(optimizer=Adam(learning_rate=0.001))

    # Custom training loop
    batch_size = 32
    epochs = 10
    train_dataset = tf.data.Dataset.from_tensor_slices((train_hidden_states, train_denoised_matrices, train_y)) \
        .batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((test_hidden_states, test_denoised_matrices, test_y)) \
        .batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

    steps_per_epoch = len(train_hidden_states) // batch_size
    validation_steps = len(test_hidden_states) // batch_size

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        for step, (x_batch, a_batch, y_batch) in enumerate(train_dataset.take(steps_per_epoch)):
            loss = model.train_step((x_batch, a_batch, y_batch))
            if step % 10 == 0:
                print(f"Step {step}: Loss = {loss['loss']:.4f}")

        # Validation
        val_losses = []
        for x_batch, a_batch, y_batch in test_dataset.take(validation_steps):
            val_loss = model.test_step((x_batch, a_batch, y_batch))
            val_losses.append(val_loss['loss'])
        print(f"Validation Loss: {tf.reduce_mean(val_losses):.4f}")

    # Make predictions
    train_outputs = model.predict([train_hidden_states, train_denoised_matrices])
    test_outputs = model.predict([test_hidden_states, test_denoised_matrices])

    return model, None, train_outputs, test_outputs
