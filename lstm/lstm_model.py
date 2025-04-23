import tensorflow as tf
from tensorflow.keras import layers, Model

def build_lstm_model(vocab_size, 
                     max_len, 
                     embed_dim=64, 
                     lstm_units=128, 
                     num_layers=2, 
                     dropout_rate=0.2, 
                     num_classes=1):
    inputs = layers.Input(shape=(max_len,), dtype=tf.int32)
    x = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim, input_length=max_len)(inputs)

    for i in range(num_layers):
        return_seq = True if i < num_layers - 1 else False
        x = layers.LSTM(units=lstm_units, return_sequences=return_seq)(x)
        if dropout_rate:
            x = layers.Dropout(dropout_rate)(x)

    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='sigmoid' if num_classes == 1 else 'softmax')(x)

    model = Model(inputs=inputs, outputs=outputs, name="LSTMClassifier")
    model.compile(optimizer='adam', 
                  loss='binary_crossentropy' if num_classes == 1 else 'sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model