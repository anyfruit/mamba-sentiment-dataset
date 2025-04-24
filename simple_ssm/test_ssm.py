# Quick test for the simple SSM model using synthetic data

import pandas as pd
import numpy as np
import tensorflow as tf
from simple_ssm_model import build_ssm_model

# Configuration
VOCAB_SIZE = 30522
SEQ_LEN = 64
NUM_SAMPLES = 200
NUM_CLASSES = 3 

# Build model
model = build_ssm_model(
    vocab_size=VOCAB_SIZE,
    max_len=SEQ_LEN,
    embed_dim=64,
    state_dim=64,
    num_layers=2,
    dropout_rate=0.1,
    num_classes=NUM_CLASSES
)
model.summary()

# Create random synthetic dataset
# inputs = np.random.randint(0, VOCAB_SIZE, size=(NUM_SAMPLES, SEQ_LEN))
# labels = np.random.randint(0, 2, size=(NUM_SAMPLES,))


# BATCH_SIZE = 8
# train_ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(100).batch(BATCH_SIZE)
# val_ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(BATCH_SIZE)

train_df = pd.read_csv("data/train_tokenized.csv")
test_df = pd.read_csv("data/test_tokenized.csv")

# Separate inputs and labels
train_inputs = train_df.drop(columns=["label"]).values
train_labels = train_df["label"].values

test_inputs = test_df.drop(columns=["label"]).values
test_labels = test_df["label"].values

BATCH_SIZE = 64

# Create TensorFlow datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)) \
                               .shuffle(buffer_size=len(train_inputs)) \
                               .batch(BATCH_SIZE) \
                               .prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels)) \
                             .batch(BATCH_SIZE) \
                             .prefetch(tf.data.AUTOTUNE)

# Train
model.fit(train_ds, validation_data=val_ds, epochs=1)

# Inference test
test_input = np.random.randint(0, VOCAB_SIZE, size=(1, SEQ_LEN))
pred = model(test_input, training=False)
print("Test prediction:", pred.numpy())