import pandas as pd
import re
from transformers import AutoTokenizer

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

# Save token ids and labels
token_ids_df = pd.DataFrame(tokenized["input_ids"])
token_ids_df["label"] = df["label"].values

# Split and save
train_df = token_ids_df.sample(frac=0.9, random_state=42)
test_df = token_ids_df.drop(train_df.index)

train_df.to_csv("train_tokenized.csv", index=False)
test_df.to_csv("test_tokenized.csv", index=False)