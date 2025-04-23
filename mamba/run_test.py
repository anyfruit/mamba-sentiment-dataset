# This file is only for simple testing for the Mamba model with synthetic data

import numpy as np
import tensorflow as tf
from config import Config
from model import build_model
from inference import predict

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
    output_dim=1,
    max_seq_len=64,
    loss_fn='binary_crossentropy',
)

# Instantiate dummy tokenizer
tokenizer = DummyTokenizer(vocab_size=args.vocab_dim, seq_len=args.max_seq_len)

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

# Train for one epoch
history = model.fit(train_dataset, validation_data=val_dataset, epochs=1)

# Inference using dummy data
fake_text = "This is a test input for Mamba"
output = predict(fake_text, model, tokenizer, args)
print("Prediction score:", output.numpy())
