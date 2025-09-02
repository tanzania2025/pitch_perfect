import os
import re
import pandas as pd
import pytest

from pitchperfect.text_sentiment_analysis.full_py_files.text_sentiment_pth import load_meld_data, clean_text, label2id

# --------------------
# Tests for clean_text
# --------------------

def test_clean_text_lowercase():
    """Text should be converted to lowercase."""
    assert clean_text("Hello WORLD!") == "hello world!"

def test_clean_text_remove_special_chars():
    """Special characters should be removed, but basic punctuation kept."""
    assert clean_text("I #love$ python!!") == "i love python!!"

def test_clean_text_strip_extra_spaces():
    """Extra whitespace should be collapsed into a single space."""
    assert clean_text("This    has   too   many   spaces") == "this has too many spaces"

def test_clean_text_nan_input():
    """NaN values should return an empty string."""
    assert clean_text(float("nan")) == ""


# -----------------------
# Tests for load_meld_data
# -----------------------

def test_load_meld_data_success(tmp_path):
    """load_meld_data should read CSVs and return DataFrames with text + label columns."""

    # Create fake train/dev/test CSVs
    for split in ["train", "dev", "test"]:
        df = pd.DataFrame({
            "Utterance": ["I am happy", "I am sad"],
            "Emotion": ["joy", "sadness"]
        })
        df.to_csv(tmp_path / f"{split}_sent_emo.csv", index=False)

    train_df, dev_df, test_df = load_meld_data(str(tmp_path))

    # Check shapes
    assert len(train_df) == 2
    assert len(dev_df) == 2
    assert len(test_df) == 2

    # Check renamed column
    assert "text" in train_df.columns
    assert "label" in train_df.columns

    # Check labels are mapped correctly
    assert set(train_df["label"]) == {label2id["joy"], label2id["sadness"]}


def test_load_meld_data_missing_file(tmp_path):
    """If a CSV is missing, the function should return (None, None, None)."""

    # Only create one CSV
    df = pd.DataFrame({
        "Utterance": ["Hi"],
        "Emotion": ["neutral"]
    })
    df.to_csv(tmp_path / "train_sent_emo.csv", index=False)

    train_df, dev_df, test_df = load_meld_data(str(tmp_path))
    assert train_df is None and dev_df is None and test_df is None
