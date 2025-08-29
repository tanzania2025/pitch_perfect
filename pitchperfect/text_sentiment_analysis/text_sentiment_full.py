#!/usr/bin/env python3
"""
MELD Emotion Classification (BiLSTM) — end‑to‑end training script

Usage:
  python meld_bilstm.py --data_dir /home/jupyter/old_backup --epochs 20

Defaults point to Vertex AI-style path /home/jupyter/old_backup containing:
  - train_sent_emo.csv
  - dev_sent_emo.csv
  - test_sent_emo.csv
"""

import os
from onnx2pytorch import ConvertModel
import pickle
import re
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import tf2onnx, onnx, torch
from onnx2pytorch import ConvertModel
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional

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
    """
    Load MELD CSVs, keep only Utterance/Emotion, and map labels.
    Returns three dataframes with columns: [text, Emotion, label].
    """
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
        print("Please ensure the MELD CSV files are in the correct directory")
        return None, None, None

    for df in (train_df, dev_df, test_df):
        df["label"] = df["Emotion"].map(label2id)
        df.dropna(subset=["label"], inplace=True)       # drop rows with unknown labels
        df.rename(columns={"Utterance": "text"}, inplace=True)

    print(f"Loaded - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    return train_df, dev_df, test_df


# ----------------------
# Text preprocessing
# ----------------------
# def clean_text(text: str) -> str:
#     """Lowercase, keep alnum + whitespace + basic punctuation, squeeze spaces."""
#     if pd.isna(text):
#         return ""
#     text = str(text).lower()
#     text = re.sub(r"[^\w\s!?.,]", " ", text)
#     text = re.sub(r"\s+", " ", text).strip()
#     return text
# preprocessing.py

def clean_text(text: str) -> str:
    """Lowercase, keep alnum + whitespace + basic punctuation, squeeze spaces."""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s!?.,]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text_columns(train_df, dev_df, test_df):
    for df in (train_df, dev_df, test_df):
        df["text_clean"] = df["text"].apply(clean_text)
    # Drop completely empty lines after cleaning
    train_df = train_df[train_df["text_clean"].str.len() > 0]
    dev_df   = dev_df[dev_df["text_clean"].str.len() > 0]
    test_df  = test_df[test_df["text_clean"].str.len() > 0]
    print(f"After cleaning - Train: {len(train_df)}, Dev: {len(dev_df)}, Test: {len(test_df)}")
    return train_df, dev_df, test_df


# ----------------------
# Tokenization utilities
# ----------------------
def tokenize_and_pad(train_df, dev_df, test_df, vocab_size=10000, max_len=128):
    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>", lower=True, split=" ")
    print("Building vocabulary...")
    tokenizer.fit_on_texts(train_df["text_clean"])
    actual_vocab_size = min(vocab_size, len(tokenizer.word_index) + 1)
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

    print(f"Final shapes - Train: {X_train_pad.shape}, Dev: {X_dev_pad.shape}, Test: {X_test_pad.shape}")
    return tokenizer, actual_vocab_size, (X_train_pad, y_train), (X_dev_pad, y_dev), (X_test_pad, y_test)


# ----------------------
# Model
# ----------------------
def create_emotion_model(vocab_size, embedding_dim=128, max_len=128, num_classes=7):
    """Create the emotion classification model (BiLSTM + Dense funnel)."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_len, mask_zero=True),
        Bidirectional(LSTM(64, return_sequences=True, dropout=0.3, recurrent_dropout=0.3)),
        Bidirectional(LSTM(32, return_sequences=False, dropout=0.3, recurrent_dropout=0.3)),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(64, activation="relu"),
        Dropout(0.4),
        Dense(32, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax")
    ])
    return model


# ----------------------
# Metrics (manual report)
# ----------------------
def classification_report_manual(y_true, y_pred, class_names):
    report = {}
    for i, name in enumerate(class_names):
        tp = np.sum((y_true == i) & (y_pred == i))
        fp = np.sum((y_true != i) & (y_pred == i))
        fn = np.sum((y_true == i) & (y_pred != i))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        support   = int(np.sum(y_true == i))
        report[name] = {"precision": precision, "recall": recall, "f1-score": f1, "support": support}
    return report


# ----------------------
# Training & evaluation
# ----------------------
def run(args):
    # Reproducibility
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # Load & preprocess
    train_df, dev_df, test_df = load_meld_data(args.data_dir)
    if train_df is None:
        raise SystemExit(1)
    train_df, dev_df, test_df = preprocess_text_columns(train_df, dev_df, test_df)

    # Tokenize
    tokenizer, vocab_size_actual, (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = tokenize_and_pad(
        train_df, dev_df, test_df, vocab_size=args.vocab_size, max_len=args.max_len
    )

    # Build model
    model = create_emotion_model(
        vocab_size=vocab_size_actual,
        embedding_dim=args.embedding_dim,
        max_len=args.max_len,
        num_classes=len(CLASSES)
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ]

    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_dev, y_dev),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\nEvaluating on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.4f}")

    # Predictions & report
    y_pred = model.predict(X_test, verbose=0)
    y_pred_labels = np.argmax(y_pred, axis=1)

    report = classification_report_manual(y_test, y_pred_labels, CLASSES)
    print("\n" + "="*60)
    print("CLASSIFICATION REPORT")
    print("="*60)
    print(f"{'Class':<12} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Support':<8}")
    print("-" * 60)
    for name, m in report.items():
        print(f"{name:<12} {m['precision']:<10.4f} {m['recall']:<10.4f} {m['f1-score']:<10.4f} {m['support']:<8d}")
    overall_accuracy = float(np.mean(y_test == y_pred_labels))
    print(f"\nOverall Accuracy: {overall_accuracy:.4f}")

def save_model_components(model, tokenizer, history, report, args, output_dir="model_outputs"):
    """
    Save model and training components to disk:
    - Keras model (.h5)
    - Tokenizer (.pkl)
    - Training history (.pkl)
    - Classification report (.pkl)
    - Training arguments (.pkl)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save Keras model
    model.save(os.path.join(output_dir, "emotion_model.h5"))

    # Save tokenizer
    with open(os.path.join(output_dir, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    # Save training history
    with open(os.path.join(output_dir, "history.pkl"), "wb") as f:
        pickle.dump(history.history, f)

    # Save classification report
    with open(os.path.join(output_dir, "report.pkl"), "wb") as f:
        pickle.dump(report, f)

    # Save training arguments
    with open(os.path.join(output_dir, "args.pkl"), "wb") as f:
        pickle.dump(vars(args), f)

    print(f"\n✅ Model components saved in {output_dir}/")

    # Optional: save model
    if args.save_pickle:
        save_model_components(
            model=model,
            tokenizer=tokenizer,
            history=history,
            report=report,
            args=args,
            output_dir=args.pickle_dir
        )

    model.save("emotion_model.h5")
    
    spec = (tf.TensorSpec((None, args.max_len), tf.int32, name="input_ids"),)
    onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)
    onnx.save(onnx_model, "emotion_model.onnx")

    pytorch_model = ConvertModel(onnx_model)
    torch.save(pytorch_model.state_dict(), "emotion_model.pth")

    print("\n✅ Models exported: emotion_model.h5 (Keras), emotion_model.onnx, emotion_model.pth (PyTorch)")

def build_argparser():
    parser = argparse.ArgumentParser(description="Train BiLSTM on MELD for 7-emotion classification.")
    parser.add_argument("--data_dir", type=str, default="/home/jupyter/old_backup", help="Directory with MELD CSVs")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Tokenizer vocabulary cap")
    parser.add_argument("--max_len", type=int, default=128, help="Max tokenized sequence length")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimensionality")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_pickle", action="store_true", help="Save model components as pickle files")
    parser.add_argument("--pickle_dir", type=str, default="model_outputs", help="Directory to save pickle files")
    return parser


if __name__ == "__main__":
    args = build_argparser().parse_args()
    run(args)
