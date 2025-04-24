# Quick test for Transformer model functionality with synthetic data

import pandas as pd
import numpy as np
import tensorflow as tf
from transformer_model import build_transformer_model
from transformers import AutoTokenizer

# Configuration
VOCAB_SIZE = 30522
SEQ_LEN = 64
NUM_SAMPLES = 200
NUM_CLASSES = 3

# Build model
model = build_transformer_model(
    vocab_size=VOCAB_SIZE,
    max_len=SEQ_LEN,
    embed_dim=64,
    num_heads=4,
    ff_dim=128,
    num_layers=2,
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
val_df = pd.read_csv("data/val_tokenized.csv")
test_df = pd.read_csv("data/test_tokenized.csv")

train_inputs = train_df.drop(columns=["label"]).values
train_labels = train_df["label"].values
val_inputs = val_df.drop(columns=["label"]).values
val_labels = val_df["label"].values
test_inputs = test_df.drop(columns=["label"]).values
test_labels = test_df["label"].values

BATCH_SIZE = 64
train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)).shuffle(len(train_inputs)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# Train
model.fit(train_ds, validation_data=val_ds, epochs=1)

# Evaluate
loss, acc = model.evaluate(test_ds)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# Inference
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
text = "This is a test input for Mamba"
tokens = tokenizer([text], max_length=SEQ_LEN, truncation=True, padding="max_length", return_tensors='tf')
pred = model(tokens["input_ids"], training=False)
class_id = tf.argmax(pred, axis=-1).numpy()[0]
sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}
print(f"Predicted Sentiment: {class_id} ({sentiment[class_id]})")

# Inference test
# test_input = np.random.randint(0, VOCAB_SIZE, size=(1, SEQ_LEN))
# pred = model(test_input, training=False)
# print("Test prediction:", pred.numpy())
