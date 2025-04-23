import tensorflow as tf
import pandas as pd
import re
from sklearn.model_selection import train_test_split


# Load data
df = pd.read_csv("sentiment.csv", encoding="latin-1", header=None)
df.columns = ["target", "ids", "date", "flag", "user", "text"]
label_map = {0: 0, 2: 1, 4: 2}
df["label"] = df["target"].map(label_map)


# Clean
def clean_tweet(tweet):
    tweet = re.sub(r"http\S+|www\S+", "", tweet)
    tweet = re.sub(r"@\w+|#\w+", "", tweet)
    tweet = re.sub(r"[^a-zA-Z\s]", "", tweet)
    return tweet.lower().strip()

df["clean_text"] = df["text"].apply(clean_tweet)
df = df[df["clean_text"].str.strip().astype(bool)]


# Split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["clean_text"], df["label"], test_size=0.1, random_state=42, stratify=df["label"]
)


# Create a TextVectorization layer
max_vocab_size = 20000
max_sequence_length = 50

vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=max_vocab_size,
    output_sequence_length=max_sequence_length,
    standardize=None
)
vectorizer.adapt(train_texts)

# Wrap text and label into TensorFlow Datasets
train_ds = tf.data.Dataset.from_tensor_slices((train_texts.values, train_labels.values))
val_ds = tf.data.Dataset.from_tensor_slices((val_texts.values, val_labels.values))

# Map texts to sequences
def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorizer(text), label

train_ds = train_ds.map(vectorize_text)
val_ds = val_ds.map(vectorize_text)

# Shuffle, batch, and prefetch
batch_size = 64
train_ds = train_ds.shuffle(10000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Save vectorizer for later reuse
input_layer = tf.keras.Input(shape=(1,), dtype=tf.string)
output_layer = vectorizer(input_layer)
vectorizer_model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

vectorizer_model.save("text_vectorizer_model")
print("TensorFlow datasets are ready!")


# Load the vectorizer model
# loaded_vectorizer_model = tf.keras.models.load_model("text_vectorizer_model")
# Extract the actual TextVectorization layer
# loaded_vectorizer = loaded_vectorizer_model.layers[1]