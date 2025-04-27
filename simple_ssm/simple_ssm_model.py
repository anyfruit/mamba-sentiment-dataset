import tensorflow as tf
from tensorflow.keras import layers, Model

class SimpleSSMLayer(layers.Layer):
    def __init__(self, state_dim, input_dim):
        super().__init__()
        self.A = self.add_weight(name="A", shape=(input_dim, state_dim), initializer="random_normal")
        self.B = self.add_weight(name="B", shape=(input_dim, state_dim), initializer="random_normal")
        self.C = self.add_weight(name="C", shape=(input_dim, state_dim), initializer="random_normal")
        self.D = self.add_weight(name="D", shape=(input_dim,), initializer="ones")

    def call(self, inputs):
        # inputs: [batch, length, dim]
        delta = tf.ones_like(inputs)
        A_bar = tf.exp(tf.cumsum(tf.einsum('bld,dn->bldn', delta, self.A), axis=1))
        B_expanded = tf.expand_dims(self.B, axis=0) # shape: [1, dim, state_dim]
        B_tiled = tf.tile(B_expanded, [tf.shape(inputs)[0], 1, 1])  # shape: [batch, dim, state_dim]
        B_tiled = tf.expand_dims(B_tiled, axis=1) # shape: [batch, 1, dim, state_dim]
        B_tiled = tf.tile(B_tiled, [1, tf.shape(inputs)[1], 1, 1]) # shape: [batch, length, dim, state_dim]
        Bu = delta[..., tf.newaxis] * inputs[..., tf.newaxis] * B_tiled  # shape: [batch, length, dim, state_dim]
        y = tf.math.cumsum(Bu * A_bar, axis=1) / (A_bar + 1e-6)
        y = tf.einsum('bldn,dn->bld', y, self.C)
        return y + inputs * self.D


def build_ssm_model(vocab_size,
                    max_len,
                    embed_dim=64,
                    state_dim=64,
                    num_layers=4,
                    dropout_rate=0.1,
                    num_classes=1):

    inputs = layers.Input(shape=(max_len,), dtype=tf.int32)
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)(inputs)

    for i in range(num_layers):
        x = SimpleSSMLayer(state_dim, embed_dim)(x)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.LayerNormalization(epsilon=1e-5)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)

    model = Model(inputs, outputs, name="SSMClassifier")
    model.compile(optimizer='adam',
                  loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
