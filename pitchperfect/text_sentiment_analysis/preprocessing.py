"""
preprocessing.py — Data loading and preprocessing for MELD

Includes:
- load_meld_data(data_dir, ...)
- clean_text(text)
- preprocess_text_columns(train_df, dev_df, test_df)
- tokenize_and_pad(train_df, dev_df, test_df, vocab_size=..., max_len=...)

Returns arrays ready for model training:
  tokenizer, actual_vocab_size, (X_train, y_train), (X_dev, y_dev), (X_test, y_test)
"""

from __future__ import annotations
from typing import Tuple, Optional
import os
import re
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Try to reuse label space from the package; otherwise fall back to defaults.
try:
    from . import CLASSES as DEFAULT_CLASSES, label2id as DEFAULT_LABEL2ID
except Exception:
    DEFAULT_CLASSES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
    DEFAULT_LABEL2ID = {c: i for i, c in enumerate(DEFAULT_CLASSES)}


# ----------------------
# Loading MELD CSV files
# ----------------------
def load_meld_data(
    data_dir: str,
    use_cols: Tuple[str, str] = ("Utterance", "Emotion"),
    label_map: Optional[dict] = None,
    verbose: bool = True,
):
    """
    Load MELD CSVs (train/dev/test), keep only needed columns, map labels, and rename text.

    Returns:
        train_df, dev_df, test_df  (each has columns: text, Emotion, label)
    """
    train_csv = os.path.join(data_dir, "train_sent_emo.csv")
    dev_csv   = os.path.join(data_dir, "dev_sent_emo.csv")
    test_csv  = os.path.join(data_dir, "test_sent_emo.csv")

    label_map = label_map or DEFAULT_LABEL2ID

    try:
        train_df = pd.read_csv(train_csv)[list(use_cols)].dropna()
        dev_df   = pd.read_csv(dev_csv)[list(use_cols)].dropna()
        test_df  = pd.read_csv(test_csv)[list(use_cols)].dropna()
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print("Please ensure the MELD CSV files are in the correct directory")
        return None, None, None

    # Map labels and rename text column
    for df in (train_df, dev_df, test_df):
        df["label"] = df["Emotion"].map(label_map)
        df.dropna(subset=["label"], inplace=True)
        df["label"] = df["label"].astype(int)
        df.rename(columns={use_cols[0]: "text"}, inplace=True)

    if verbose:
        print(f"Loaded - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
        _print_label_distribution(train_df, title="Training")

    return train_df, dev_df, test_df


def _print_label_distribution(df: pd.DataFrame, title: str = "Dataset") -> None:
    print(f"\nLabel distribution in {title} data:")
    counts = df["Emotion"].value_counts()
    for emotion, count in counts.items():
        print(f"  {emotion}: {count}")


# ----------------------
# Text preprocessing
# ----------------------
def clean_text(text: str) -> str:
    """
    Lowercase; keep alnum, whitespace, and basic punctuation (emotion-relevant: ! ? . ,).
    Squeeze multiple spaces.
    """
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text_columns(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    verbose: bool = True
):
    """
    Adds 'text_clean' to each df, drops empty rows after cleaning.
    """
    for df in (train_df, dev_df, test_df):
        df["text_clean"] = df["text"].apply(clean_text)

    train_df = train_df[train_df["text_clean"].str.len() > 0]
    dev_df   = dev_df[dev_df["text_clean"].str.len() > 0]
    test_df  = test_df[test_df["text_clean"].str.len() > 0]

    if verbose:
        print(f"After cleaning - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")

    return train_df, dev_df, test_df


# ----------------------
# Tokenization & padding
# ----------------------
def tokenize_and_pad(
    train_df: pd.DataFrame,
    dev_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vocab_size: int = 10000,
    max_len: int = 128,
    verbose: bool = True,
):
    """
    Builds a Tokenizer on train 'text_clean', converts to sequences, and pads.
    Returns:
        tokenizer,
        actual_vocab_size,
        (X_train, y_train),
        (X_dev, y_dev),
        (X_test, y_test)
    """
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", lower=True, split=" ")

    if verbose:
        print("Building vocabulary...")

    tokenizer.fit_on_texts(train_df["text_clean"])
    actual_vocab_size = min(vocab_size, len(tokenizer.word_index) + 1)

    if verbose:
        print(f"Vocabulary size (actual): {actual_vocab_size} (limit: {vocab_size})")

    X_train = tokenizer.texts_to_sequences(train_df["text_clean"])
    X_dev   = tokenizer.texts_to_sequences(dev_df["text_clean"])
    X_test  = tokenizer.texts_to_sequences(test_df["text_clean"])

    X_train_pad = pad_sequences(X_train, maxlen=max_len, padding="post", truncating="post")
    X_dev_pad   = pad_sequences(X_dev,   maxlen=max_len, padding="post", truncating="post")
    X_test_pad  = pad_sequences(X_test,  maxlen=max_len, padding="post", truncating="post")

    y_train = train_df["label"].values.astype("int64")
    y_dev   = dev_df["label"].values.astype("int64")
    y_test  = test_df["label"].values.astype("int64")

    if verbose:
        print(f"Final shapes - Train: {X_train_pad.shape}, Dev: {X_dev_pad.shape}, Test: {X_test_pad.shape}")

    return tokenizer, actual_vocab_size, (X_train_pad, y_train), (X_dev_pad, y_dev), (X_test_pad, y_test)
