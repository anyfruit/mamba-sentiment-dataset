# Quick test for LSTM model functionality with synthetic data

import numpy as np
import tensorflow as tf
from lstm_model import build_lstm_model

# Configuration
VOCAB_SIZE = 30522
SEQ_LEN = 64
NUM_SAMPLES = 200
NUM_CLASSES = 1  

# Build model
model = build_lstm_model(
    vocab_size=VOCAB_SIZE,
    max_len=SEQ_LEN,
    embed_dim=64,
    lstm_units=128,
    num_layers=2,
    dropout_rate=0.2,
    num_classes=NUM_CLASSES
)
model.summary()

# Create random synthetic dataset
inputs = np.random.randint(0, VOCAB_SIZE, size=(NUM_SAMPLES, SEQ_LEN))
labels = np.random.randint(0, 2, size=(NUM_SAMPLES,))


BATCH_SIZE = 8
train_ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(100).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(BATCH_SIZE)

# Train
model.fit(train_ds, validation_data=val_ds, epochs=1)

# Inference test
test_input = np.random.randint(0, VOCAB_SIZE, size=(1, SEQ_LEN))
pred = model(test_input, training=False)
print("Test prediction:", pred.numpy())
