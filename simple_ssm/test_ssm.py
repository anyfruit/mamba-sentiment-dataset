import time
import csv
import json
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from simple_ssm_model import build_ssm_model
from transformers import AutoTokenizer

# Configuration
VOCAB_SIZE = 30522
SEQ_LEN = 64
NUM_SAMPLES = 200
NUM_CLASSES = 1  
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
inputs = np.random.randint(0, VOCAB_SIZE, size=(NUM_SAMPLES, SEQ_LEN))
labels = np.random.randint(0, 2, size=(NUM_SAMPLES,))


BATCH_SIZE = 8
train_ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(100).batch(BATCH_SIZE)
val_ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(BATCH_SIZE)
# inputs = np.random.randint(0, VOCAB_SIZE, size=(NUM_SAMPLES, SEQ_LEN))
# labels = np.random.randint(0, 2, size=(NUM_SAMPLES,))


# BATCH_SIZE = 8
# train_ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).shuffle(100).batch(BATCH_SIZE)
# val_ds = tf.data.Dataset.from_tensor_slices((inputs, labels)).batch(BATCH_SIZE)

# Load data
train_df = pd.read_csv("data/train_tokenized.csv")
val_df = pd.read_csv("data/val_tokenized.csv")
test_df = pd.read_csv("data/test_tokenized.csv")

# CHANGED BELOW BLOCK FOR SUBSET TESTING 
train_inputs = train_df.drop(columns=["label"]).values
train_labels = train_df["label"].values
val_inputs = val_df.drop(columns=["label"]).values
val_labels = val_df["label"].values
test_inputs = test_df.drop(columns=["label"]).values
test_labels = test_df["label"].values

# SUBSET TESTING
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
train_ds = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels)).shuffle(len(train_inputs)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
val_ds = tf.data.Dataset.from_tensor_slices((val_inputs, val_labels)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_ds = tf.data.Dataset.from_tensor_slices((test_inputs, test_labels)).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Train
model.fit(train_ds, validation_data=val_ds, epochs=1)

# Evaluate
loss, acc = model.evaluate(test_ds)
print(f"\nTest Loss: {loss:.4f}, Test Accuracy: {acc:.4f}")

# Inference test
# tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# text = "This is a test input for Mamba"
# tokens = tokenizer([text], max_length=SEQ_LEN, truncation=True, padding="max_length", return_tensors='tf')
# pred = model(tokens["input_ids"], training=False)
# class_id = tf.argmax(pred, axis=-1).numpy()[0]
# sentiment = {0: "Negative", 1: "Neutral", 2: "Positive"}
# print(f"Predicted Sentiment: {class_id} ({sentiment[class_id]})")

# Inference test
# test_input = np.random.randint(0, VOCAB_SIZE, size=(1, SEQ_LEN))
# pred = model(test_input, training=False)
# print("Test prediction:", pred.numpy())

# === Inference loop over test set ===
os.makedirs("results", exist_ok=True)
inference_log_path = "results/ssm_inference_log.csv"
summary_path = "results/ssm_summary.json"

with open(inference_log_path, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["tweet_id", "seq_len", "runtime_s"])

    correct = 0
    total = 0
    runtimes = []

    for i, (x, y_true) in enumerate(zip(test_inputs, test_labels)):
        seq_len = np.count_nonzero(x)

        # Inference and timing
        x_input = np.expand_dims(x, axis=0)
        start_time = time.time()
        pred = model(x_input, training=False).numpy()
        runtime = time.time() - start_time
        runtimes.append(runtime)

        pred_class = np.argmax(pred, axis=-1).item()
        if pred_class == y_true:
            correct += 1
        total += 1
        writer.writerow([i + 1, seq_len, round(runtime, 5)])

# Save summary
summary = {
    "model": "SSM",
    "accuracy": round(correct / total, 4),
    "avg_inference_time": round(np.mean(runtimes), 5),
    "total_samples": total
}

with open(summary_path, "w") as f:
    json.dump(summary, f, indent=2)

print(f"\nInference logging complete for SSM")
print(f"Accuracy: {summary['accuracy']}, Avg runtime: {summary['avg_inference_time']}s")