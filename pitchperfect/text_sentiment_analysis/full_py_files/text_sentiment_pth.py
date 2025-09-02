#!/usr/bin/env python3
"""
MELD Emotion Classification (BiLSTM) — Pure PyTorch implementation
Exports ONLY emotionmodel.pth file (no other files)
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
from sklearn.metrics import accuracy_score

# ----------------------
# Constants & label maps
# ----------------------

# CLASSES: list of the 7 emotions you want to predict.
# label2id: dictionary mapping each emotion to an integer ID.

CLASSES = ["anger","disgust","fear","joy","neutral","sadness","surprise"]
label2id = {c: i for i, c in enumerate(CLASSES)}
id2label = {i: c for c, i in label2id.items()}

# ----------------------
# Simple tokenizer class
# ----------------------

# <PAD> = 0 → for padding short sentences.
# <OOV> = 1 → for words not seen in training.

class SimpleTokenizer:
    def __init__(self, vocab_size=10000):
        self.vocab_size = vocab_size
        self.word2idx = {"<PAD>": 0, "<OOV>": 1}
        self.idx2word = {0: "<PAD>", 1: "<OOV>"}
        
# Reads through training texts, counts word frequencies.
# Sorts words by frequency and keeps the top ones (vocab_size - 2).
# Assigns IDs starting at 2, since 0 and 1 are already taken.

    def fit(self, texts):
        word_freq = {}
        for text in texts:
            for word in text.split():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Sort by frequency and take top vocab_size-2 (excluding PAD, OOV)
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        for i, (word, _) in enumerate(sorted_words[:self.vocab_size-2]):
            idx = i + 2
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
# Splits a new sentence into words.
# Looks up each word in word2idx.
# If missing → maps to 1 (<OOV>).
# Pads with 0s (<PAD>) or truncates to max_len.
    
    def encode(self, text, max_len):
        tokens = [self.word2idx.get(word, 1) for word in text.split()]  # 1 = OOV
        if len(tokens) > max_len:
            tokens = tokens[:max_len]
        else:
            tokens += [0] * (max_len - len(tokens))  # 0 = PAD
        return tokens

# ----------------------
# Dataset class
# ----------------------
# Takes:
# texts: list of raw sentences
# labels: list of integer labels (e.g., [3, 5, 2])
# tokenizer: instance of SimpleTokenizer
# max_len: fixed length for sequences
# For each (text, label) pair:
# Encodes the text into a sequence of integers with padding/truncation.
# Converts both the sequence and label into PyTorch tensors of type long (required for embeddings & loss functions like CrossEntropy).
# Stores them in self.data as a tuple (tokens_tensor, label_tensor).

class MELDDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.data = []
        for text, label in zip(texts, labels):
            tokens = tokenizer.encode(text, max_len)
            self.data.append((torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)))
            
# Returns how many samples the dataset has.
# Needed by PyTorch’s DataLoader.

    def __len__(self):
        return len(self.data)
    
# Returns the (tokens_tensor, label_tensor) pair at position idx.
# This is what DataLoader calls when creating batches.

    def __getitem__(self, idx):
        return self.data[idx]

# ----------------------
# BiLSTM Model (Pure PyTorch)
# ----------------------
# This class defines a bi-directional LSTM-based neural network for emotion classification.

# Inherits from nn.Module (all PyTorch models do).
# Parameters:
# vocab_size = number of words in tokenizer vocab.
# embedding_dim=128 = size of word vectors.
# hidden_dim1=64, hidden_dim2=32 = hidden sizes of first and second LSTM layers.
# num_classes=7 = emotions (anger, joy, etc.).

class BiLSTMEmotionClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim1=64, hidden_dim2=32, num_classes=7):
        super(BiLSTMEmotionClassifier, self).__init__()
        
        # Turns word IDs → vectors of length 128.
        # padding_idx=0 ensures padding tokens always map to a zero vector.
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # First BiLSTM layer
        # Processes embeddings with a bi-directional LSTM (forward + backward context).
        # Output size = hidden_dim1 * 2 (because of bidirectionality).
        self.bilstm1 = nn.LSTM(embedding_dim, hidden_dim1, batch_first=True, bidirectional=True, dropout=0.3)
        
        # Second BiLSTM layer
        # Input = output of the first BiLSTM.
        # Output hidden size = hidden_dim2 * 2.
        self.bilstm2 = nn.LSTM(hidden_dim1*2, hidden_dim2, batch_first=True, bidirectional=True, dropout=0.3)
        
        # Fully connected (dense) layers
        # A stack of dense layers gradually reduces feature size.
        # Dropout prevents overfitting.
        # Final output = logits of size [batch_size, num_classes].
        self.fc1 = nn.Linear(hidden_dim2*2, 128)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(64, 32)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(32, num_classes)
    
    # Forward pass
    # Extracts final forward and backward hidden states (h_n[-2], h_n[-1]).
    # Passes through dense layers → classification logits.
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embedding_dim)
        
        # First BiLSTM layer
        lstm1_out, _ = self.bilstm1(embedded)  # (batch_size, seq_len, hidden_dim1*2)
        
        # Second BiLSTM layer
        lstm2_out, (h_n, _) = self.bilstm2(lstm1_out)  # h_n: (4, batch_size, hidden_dim2)
        
        # Concatenate final forward and backward hidden states
        final_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # (batch_size, hidden_dim2*2)
        
        # Dense layers
        # A stack of linear layers reduces feature size:
        # hidden_dim2*2 → 128 → 64 → 32 → num_classes.
        # Dropout prevents overfitting.
        # Final layer outputs logits = raw scores for each class.
        out = torch.relu(self.fc1(final_hidden))
        out = self.dropout1(out)
        out = torch.relu(self.fc2(out))
        out = self.dropout2(out)
        out = torch.relu(self.fc3(out))
        out = self.dropout3(out)
        out = self.fc4(out)
        
        return out

# ----------------------
# Data loading
# ----------------------
def load_meld_data(data_dir: str):
    train_csv = os.path.join(data_dir, "train_sent_emo.csv")
    dev_csv   = os.path.join(data_dir, "dev_sent_emo.csv")
    test_csv  = os.path.join(data_dir, "test_sent_emo.csv")
    
    use_cols = ["Utterance", "Emotion"]
    try:
        train_df = pd.read_csv(train_csv)[use_cols].dropna()
        dev_df   = pd.read_csv(dev_csv)[use_cols].dropna()
        test_df  = pd.read_csv(test_csv)[use_cols].dropna()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return None, None, None
    
    for df in (train_df, dev_df, test_df):
        df["label"] = df["Emotion"].map(label2id)
        df.dropna(subset=["label"], inplace=True)
        df.rename(columns={"Utterance": "text"}, inplace=True)
    
    print(f"Loaded - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    return train_df, dev_df, test_df

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ----------------------
# Training & evaluation
# ----------------------
def run(args):
    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load & clean data
    train_df, dev_df, test_df = load_meld_data(args.data_dir)
    if train_df is None:
        raise SystemExit(1)
    
    for df in (train_df, dev_df, test_df):
        df["text_clean"] = df["text"].apply(clean_text)
        df = df[df["text_clean"].str.len() > 0]
    
    # Build tokenizer
    tokenizer = SimpleTokenizer(vocab_size=args.vocab_size)
    tokenizer.fit(train_df["text_clean"].tolist())
    
    # Create datasets
    train_ds = MELDDataset(train_df["text_clean"].tolist(), train_df["label"].values, tokenizer, args.max_len)
    dev_ds = MELDDataset(dev_df["text_clean"].tolist(), dev_df["label"].values, tokenizer, args.max_len)
    test_ds = MELDDataset(test_df["text_clean"].tolist(), test_df["label"].values, tokenizer, args.max_len)
    
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    dev_dl = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False)
    test_dl = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    
    # Model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BiLSTMEmotionClassifier(
        vocab_size=len(tokenizer.word2idx),
        embedding_dim=args.embedding_dim,
        num_classes=len(CLASSES)
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Training loop
    best_val_acc = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_x, batch_y in train_dl:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_y.size(0)
            train_correct += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in dev_dl:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()
        
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Acc: {train_acc:.2f}% - Val Acc: {val_acc:.2f}%")
        
        # Early stopping logic
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model state
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered!")
                break
        
        scheduler.step(val_acc)
    
    # Load best model weights
    model.load_state_dict(best_model_state)
    
    # Final test evaluation
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_dl:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            test_preds.extend(predicted.cpu().numpy())
            test_labels.extend(batch_y.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # ✅ SAVE ONLY THE .PTH FILE
    print("\n" + "="*60)
    print("SAVING PYTORCH MODEL")
    print("="*60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, "emotionmodel.pth")
    
    # Save the complete model (architecture + weights)
    torch.save(model, model_path)
    print(f"🎉 PyTorch model saved as: {model_path}")
    print("Export complete! Only emotionmodel.pth was created.")

def build_argparser():
    parser = argparse.ArgumentParser(description="Train BiLSTM on MELD and export ONLY emotionmodel.pth")
    parser.add_argument("--data_dir", type=str, default="/home/jupyter/old_backup", help="Directory with MELD CSVs")
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save emotionmodel.pth")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Tokenizer vocabulary cap")
    parser.add_argument("--max_len", type=int, default=128, help="Max sequence length")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimensionality")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    return parser

if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)