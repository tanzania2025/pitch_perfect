# utils_data.py
import os
import pandas as pd
from datasets import Dataset, DatasetDict
from collections import Counter
import torch

CLASSES = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
label2id = {c: i for i, c in enumerate(CLASSES)}
id2label = {i: c for c, i in label2id.items()}


def load_meld_splits(
    data_dir,
    train_csv="train_sent_emo.csv",
    dev_csv="dev_sent_emo.csv",
    test_csv="test_sent_emo.csv",
):
    use_cols = ["Utterance", "Emotion"]

    def _read(fn):
        df = pd.read_csv(os.path.join(data_dir, fn))[use_cols].dropna()
        df = df[df.Emotion.isin(CLASSES)].copy()
        df["label"] = df["Emotion"].map(label2id)
        df.rename(columns={"Utterance": "text"}, inplace=True)
        return df[["text", "label"]].reset_index(drop=True)

    train_df = _read(train_csv)
    dev_df = _read(dev_csv)
    test_df = _read(test_csv)
    return train_df, dev_df, test_df


def make_hf_datasets(train_df, dev_df, test_df):
    return DatasetDict(
        {
            "train": Dataset.from_pandas(train_df),
            "validation": Dataset.from_pandas(dev_df),
            "test": Dataset.from_pandas(test_df),
        }
    )


def tokenize_datasets(ds, tokenizer, max_length=128):
    def tok(batch):
        return tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=max_length
        )

    keep_cols = ["label"]
    return ds.map(
        tok,
        batched=True,
        remove_columns=[c for c in ds["train"].column_names if c not in keep_cols],
    )


def compute_class_weights(train_df):
    counts = Counter(train_df["label"].tolist())
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(CLASSES) * counts[i]) for i in range(len(CLASSES))],
        dtype=torch.float,
    )
    return weights
