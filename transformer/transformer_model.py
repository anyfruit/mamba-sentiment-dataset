import tensorflow as tf
from tensorflow.keras import layers, Model


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) # Self-attention to compute token-token dependencies
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ]) # Feedforward sublayer projects to a larger hidden space ff_dim
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs) # [batch, seq_len, embed_dim]
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output) # Residual connection
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    


class PositionalEmbedding(layers.Layer): # Combine token and position information
    def __init__(self, sequence_length, vocab_size, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=sequence_length, output_dim=embed_dim)

    def call(self, x):
        positions = tf.range(start=0, limit=tf.shape(x)[-1], delta=1)
        embedded_tokens = self.token_emb(x)
        embedded_positions = self.pos_emb(positions)
        return embedded_tokens + embedded_positions # [batch, seq_len, embedd_dim]
    



def build_transformer_model(vocab_size, 
                            max_len, 
                            embed_dim=64, 
                            num_heads=4, 
                            ff_dim=128, 
                            num_layers=4,
                            dropout_rate=0.1,
                            num_classes=1):
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32)
    x = PositionalEmbedding(max_len, vocab_size, embed_dim)(inputs)

    for _ in range(num_layers):
        x = TransformerBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)

    model = Model(inputs, outputs, name='TransformerClassifier')
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
                  metrics = ['accuracy'])
    
    return model
