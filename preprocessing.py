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