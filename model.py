# train.py

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import pickle
import re
from collections import Counter
from sklearn.model_selection import train_test_split
from model import SentimentRNN

MAX_VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 5
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    return text

def build_vocab(texts):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in counter.most_common(MAX_VOCAB_SIZE - 2):
        vocab[word] = len(vocab)

    return vocab

def encode_text(text, vocab):
    tokens = text.split()
    encoded = [vocab.get(t, vocab['<UNK>']) for t in tokens]
    return encoded[:MAX_LEN] + [0]*(MAX_LEN - len(encoded))

# Load data
data = pd.read_csv("reviews.csv")
data["review"] = data["review"].astype(str).apply(clean_text)

texts = data["review"].tolist()
labels = torch.tensor(data["sentiment"].values, dtype=torch.float32)

vocab = build_vocab(texts)

# Save vocab
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

X_encoded = [encode_text(t, vocab) for t in texts]
X_tensor = torch.tensor(X_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X_tensor, labels, test_size=0.2, random_state=42
)

model = SentimentRNN(len(vocab)).to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train.to(device)).squeeze()
    loss = criterion(outputs, y_train.to(device))
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# Save model
torch.save(model.state_dict(), "sentiment_model.pth")

print("✅ Model & vocab saved successfully!")