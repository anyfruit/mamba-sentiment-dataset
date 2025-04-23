import tensorflow as tf
from tensorflow.keras import layers
from einops import rearrange, repeat
import numpy as np
from ssm import dynamic_state_update

class SelectiveBlock(layers.Layer):
    # Should include:
    # Input projection
    # Convolutional mixing
    # SSM (already defined)
    # Output projection

    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        self.linear_in = layers.Dense(config.hidden_dim * 2, use_bias=False)
        self.local_conv = layers.Conv1D(
            filters=config.hidden_dim,
            kernel_size=config.kernel_size,
            use_bias=config.conv_bias,
            padding='causal' # No future tokens influence current ones
        )
        self.param_proj = layers.Dense(config.rank_delta + config.state_dim * 2, use_bias=False)
        self.delta_proj = layers.Dense(config.hidden_dim, use_bias=True)

        base_vals = tf.range(1, config.state_dim + 1, dtype=tf.float32)
        self.log_A = tf.Variable(tf.math.log(repeat(base_vals, 'n -> d n', d=config.hidden_dim)), trainable=True)
        self.residual_gain = tf.Variable(np.ones(config.hidden_dim), dtype=tf.float32)
        self.out_proj = layers.Dense(config.embed_dim, use_bias=config.dense_bias)


    def call(self, inputs):
        x_proj = self.linear_in(inputs)
        stream, res = tf.split(x_proj, num_or_size_splits=2, axis=-1)
        stream = rearrange(stream, 'b l d -> b d l')
        stream = self.local_conv(stream)[:, :, :tf.shape(inputs)[1]]
        stream = rearrange(stream, 'b d l -> b l d')
        stream = tf.nn.swish(stream) # Swish activation used in SSM
        result = self.run_ssm(stream) * tf.nn.swish(res)

        return self.out_proj(result)
    

    def run_ssm(self, sequence):
        A = -tf.exp(self.log_A)
        BCD = self.param_proj(sequence)
        delta, B, C = tf.split(BCD, [self.config.rank_delta, self.config.state_dim, self.config.state_dim], axis=-1)
        delta = tf.nn.softplus(self.delta_proj(delta))
        
        return dynamic_state_update(sequence, delta, A, B, C, self.residual_gain)