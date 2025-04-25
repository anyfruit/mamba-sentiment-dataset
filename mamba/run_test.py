import time
import csv
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from config import Config
from model import build_model
from inference import predict
from transformers import AutoTokenizer


# Dummy tokenizer for testing
class DummyTokenizer:
    def __init__(self, vocab_size, seq_len):
        self.vocab_size = vocab_size
        self.seq_len = seq_len

    def __call__(self, texts, **kwargs):
        batch_size = len(texts) if isinstance(texts, list) else 1
        return {'input_ids': np.random.randint(0, self.vocab_size, size=(batch_size, self.seq_len))}

# Define config 
args = Config(
    embed_dim=64,
    state_dim=16,
    depth=2,
    dropout_prob=0.1,
    vocab_dim=30522,  
#    output_dim=1,
    max_seq_len=64,
#    loss_fn='binary_crossentropy',
    output_dim=3,
    loss_fn='sparse_categorical_crossentropy',
)

# Instantiate dummy tokenizer
# tokenizer = DummyTokenizer(vocab_size=args.vocab_dim, seq_len=args.max_seq_len)

# Instantiate tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

# Build model
model, _ = build_model(args)
model.summary()

# Create random synthetic dataset
NUM_SAMPLES = 200
BATCH_SIZE = 8

# Random integer token IDs and labels
random_inputs = np.random.randint(0, args.vocab_dim, size=(NUM_SAMPLES, args.max_seq_len))
random_labels = np.random.randint(0, 2, size=(NUM_SAMPLES,))

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((random_inputs, random_labels)).shuffle(100).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices((random_inputs, random_labels)).batch(BATCH_SIZE)
# # Random integer token IDs and labels
# random_inputs = np.random.randint(0, args.vocab_dim, size=(NUM_SAMPLES, args.max_seq_len))
# random_labels = np.random.randint(0, 2, size=(NUM_SAMPLES,))

# # Create TensorFlow datasets
# train_dataset = tf.data.Dataset.from_tensor_slices((random_inputs, random_labels)).shuffle(100).batch(BATCH_SIZE)
# val_dataset = tf.data.Dataset.from_tensor_slices((random_inputs, random_labels)).batch(BATCH_SIZE)


train_df = pd.read_csv("data/train_tokenized.csv")
test_df = pd.read_csv("data/test_tokenized.csv")
val_df = pd.read_csv("data/val_tokenized.csv")

# CHANGE BELOW BLOCK FOR SUBSET TESTING
train_inputs = train_df.drop(columns=["label"]).values
train_labels = train_df["label"].values

test_inputs = test_df.drop(columns=["label"]).values
test_labels = test_df["label"].values

val_inputs = val_df.drop(columns=["label"]).values
val_labels = val_df["label"].values

#SUBSET TESTING
# subset_frac = 0.01

# train_df_subset = train_df.sample(frac=subset_frac, random_state=42)
# val_df_subset = val_df.sample(frac=subset_frac, random_state=42)
# test_df_subset = test_df.sample(frac=subset_frac, random_state=42)

# train_inputs = train_df_subset.drop(columns=["label"]).values
# train_labels = train_df_subset["label"].values

# val_inputs = val_df_subset.drop(columns=["label"]).values
# val_labels = val_df_subset["label"].values

# test_inputs = test_df_subset.drop(columns=["label"]).values
# test_labels = test_df_subset["label"].values

BATCH_SIZE = 64

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)) \
                               .shuffle(buffer_size=len(train_inputs)) \
                               .batch(BATCH_SIZE) \
                               .prefetch(tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels)) \
                             .batch(BATCH_SIZE) \
                             .prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels)) \
                             .batch(BATCH_SIZE) \
                             .prefetch(tf.data.AUTOTUNE)


# Train for one epoch
history = model.fit(train_dataset, validation_data=val_dataset, epochs=1)


# Evaluate on test set
loss, acc = model.evaluate(test_dataset)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# sample_text = "This is a test input for Mamba"
# pred_output = predict(sample_text, model, tokenizer, args)
# pred_class = tf.argmax(pred_output, axis=-1).numpy().item()

# # Interpret prediction
# sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
# print(f"Predicted Sentiment: {pred_class} ({sentiment_map[pred_class]})")

os.makedirs("results", exist_ok=True)
inference_log_path = "results/mamba_inference_log.csv"
summary_path = "results/mamba_summary.json"

with open(inference_log_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["tweet_id", "seq_len", "runtime_s"])

    correct = 0
    total = 0
    runtimes = []

    for i, (x, y_true) in enumerate(zip(test_inputs, test_labels)):
        # Sequence length = count of non-zero tokens (ignore padding)
        seq_len = np.count_nonzero(x)

        # Predict and time
        x_input = np.expand_dims(x, axis=0)
        start_time = time.time()
        pred_probs = model.predict(x_input, verbose=0)
        runtime = time.time() - start_time
        runtimes.append(runtime)

        # Save prediction metadata
        pred_class = np.argmax(pred_probs, axis=-1).item()
        if pred_class == y_true:
            correct += 1
        total += 1
        writer.writerow([i + 1, seq_len, round(runtime, 5)])

# Save summary
summary = {
    "model": "Mamba",
    "accuracy": round(correct / total, 4),
    "avg_inference_time": round(np.mean(runtimes), 5),
    "total_samples": total
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nLogged inference for {total} tweets")
print(f"Accuracy: {summary['accuracy']}, Avg runtime: {summary['avg_inference_time']}s")