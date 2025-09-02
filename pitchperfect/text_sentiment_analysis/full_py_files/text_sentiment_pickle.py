#!/usr/bin/env python3
"""
MELD Emotion Classification (BiLSTM, PyTorch implementation)
Exports trained model directly to PyTorch format (.pth)
"""

import os
import re
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ----------------------
# Constants & label maps
# ----------------------
CLASSES = ["anger","disgust","fear","joy","neutral","sadness","surprise"]
label2id = {c: i for i, c in enumerate(CLASSES)}
id2label = {i: c for c, i in label2id.items()}

# ----------------------
# Data loading utilities
# ----------------------
def load_meld_data(data_dir: str):
    use_cols = ["Utterance", "Emotion"]
    train_df = pd.read_csv(os.path.join(data_dir, "train_sent_emo.csv"))[use_cols].dropna()
    dev_df   = pd.read_csv(os.path.join(data_dir, "dev_sent_emo.csv"))[use_cols].dropna()
    test_df  = pd.read_csv(os.path.join(data_dir, "test_sent_emo.csv"))[use_cols].dropna()

    for df in (train_df, dev_df, test_df):
        df["label"] = df["Emotion"].map(label2id)
        df.rename(columns={"Utterance": "text"}, inplace=True)

    return train_df, dev_df, test_df

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------
# Dataset class
# ----------------------
class MELDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.sequences = tokenizer.texts_to_sequences(texts)
        self.labels = labels
        self.max_len = max_len

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        length = torch.tensor(len(seq), dtype=torch.long)
        return seq, label, length

def collate_fn(batch):
    sequences, labels, lengths = zip(*batch)
    lengths = torch.tensor([min(len(seq), max(lengths)) for seq in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded = padded[:, :max(lengths)]  # trim to max actual length
    return padded, torch.stack(labels), lengths

# ----------------------
# Model (PyTorch BiLSTM)
# ----------------------
class BiLSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim1=64, hidden_dim2=32, num_classes=7, pad_idx=0):
        super(BiLSTMEmotionClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.bilstm1 = nn.LSTM(embedding_dim, hidden_dim1, batch_first=True, bidirectional=True, dropout=0.3)
        self.bilstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, batch_first=True, bidirectional=True, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(hidden_dim2*2, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_classes)

    def forward(self, x, lengths):
        embedded = self.embedding(x)
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.bilstm1(packed)
        _, (h_n2, _) = self.bilstm2(packed)
        out = torch.cat((h_n2[-2], h_n2[-1]), dim=1)
        out = self.dropout(torch.relu(self.fc1(out)))
        out = self.dropout(torch.relu(self.fc2(out)))
        out = self.dropout(torch.relu(self.fc3(out)))
        return self.fc4(out)

# ----------------------
# Training & evaluation
# ----------------------
def run(args):
    # Load data
    train_df, dev_df, test_df = load_meld_data(args.data_dir)
    for df in (train_df, dev_df, test_df):
        df["text"] = df["text"].apply(clean_text)

    # Tokenizer
    tokenizer = Tokenizer(num_words=args.vocab_size, oov_token="<OOV>", lower=True)
    tokenizer.fit_on_texts(train_df["text"])
    vocab_size_actual = min(args.vocab_size, len(tokenizer.word_index) + 1)

    # Build datasets
    train_ds = MELDDataset(train_df["text"].tolist(), train_df["label"].values, tokenizer, args.max_len)
    dev_ds   = MELDDataset(dev_df["text"].tolist(), dev_df["label"].values, tokenizer, args.max_len)
    test_ds  = MELDDataset(test_df["text"].tolist(), test_df["label"].values, tokenizer, args.max_len)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_dl   = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_dl  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BiLSTMEmotionClassifier(vocab_size_actual, args.embedding_dim, num_classes=len(CLASSES)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for X_batch, y_batch, lengths in train_dl:
            X_batch, y_batch, lengths = X_batch.to(device), y_batch.to(device), lengths.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch, lengths)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(train_dl):.4f}")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch, lengths in test_dl:
            X_batch, y_batch, lengths = X_batch.to(device), y_batch.to(device), lengths.to(device)
            outputs = model(X_batch, lengths)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    print(f"\nTest Accuracy: {acc:.4f}")

    # ✅ Save the full PyTorch model
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "emotion_model.pth")
    torch.save(model, model_path)
    print(f"🎉 Saved PyTorch model to {model_path}")

# ----------------------
# Argparser
# ----------------------
def build_argparser():
    parser = argparse.ArgumentParser(description="Train BiLSTM on MELD and export to PyTorch (.pth)")
    parser.add_argument("--data_dir", type=str, default="/home/jupyter/old_backup")
    parser.add_argument("--output_dir", type=str, default="../../pytorch_models")
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--embedding_dim", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
