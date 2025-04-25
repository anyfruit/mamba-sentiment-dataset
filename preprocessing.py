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
import pandas as pd
import re
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

# Load the dataset
df = pd.read_csv("data/sentiment.csv", encoding="latin-1", header=None)
df.columns = ["target", "ids", "date", "flag", "user", "text"]

# Map sentiment to 0 (negative), 1 (neutral), 2 (positive)
label_map = {0: 0, 2: 1, 4: 2}
df["label"] = df["target"].map(label_map)

# Clean tweet text
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
# Tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
max_length = 64

# Tokenize
tokenized = tokenizer(
    df["clean_text"].tolist(),
    max_length=max_length,
    truncation=True,
    padding="max_length",
    return_tensors="np"
)

# Combine into DataFrame
token_ids_df = pd.DataFrame(tokenized["input_ids"])
token_ids_df["label"] = df["label"].values

# Split into 98% train, 1% val, 1% test
train_df, temp_df = train_test_split(token_ids_df, test_size=0.02, stratify=token_ids_df["label"], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42)

# Save
train_df.to_csv("data/train_tokenized.csv", index=False)
val_df.to_csv("data/val_tokenized.csv", index=False)
test_df.to_csv("data/test_tokenized.csv", index=False)
