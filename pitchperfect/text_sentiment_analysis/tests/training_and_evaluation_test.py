import os
import pandas as pd
import torch
import pytest

from pitchperfect.text_sentiment_analysis.full_py_files.text_sentiment_pth import run, build_argparser

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def make_fake_meld_csvs(tmp_path):
    """Create minimal fake MELD train/dev/test CSVs."""
    for split in ["train", "dev", "test"]:
        df = pd.DataFrame({
            "Utterance": ["I am happy", "I am sad"],
            "Emotion": ["joy", "sadness"]
        })
        df.to_csv(tmp_path / f"{split}_sent_emo.csv", index=False)
    return tmp_path


# -------------------------------------------------
# Tests
# -------------------------------------------------

def test_run_missing_data(tmp_path):
    """If data_dir is missing required CSVs, run() should exit with SystemExit."""
    args = build_argparser().parse_args([])
    args.data_dir = str(tmp_path)  # empty dir
    args.output_dir = str(tmp_path)
    args.epochs = 1  # keep runtime short

    with pytest.raises(SystemExit):
        run(args)

MODEL_PATH = "pitchperfect/text_sentiment_analysis/final_model/text_sentiment_model.pth"

def test_model_file_exists():
    """Check that the final trained model file exists in the expected location."""
    assert os.path.exists(MODEL_PATH), f"Model file not found at {MODEL_PATH}"