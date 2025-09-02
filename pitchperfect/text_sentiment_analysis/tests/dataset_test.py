import torch
import pytest
from pitchperfect.text_sentiment_analysis.full_py_files.text_sentiment_pth import SimpleTokenizer, MELDDataset

@pytest.fixture
def sample_dataset():
    """
    The constructor builds a mini dataset:
    2 sentences, tokenized and padded.
    2 labels, wrapped as tensors.
    Returns a MELDDataset object so your tests can check its behavior.
    """
    texts = ["i love pears", "pears are great"]
    labels = [3, 4]  # e.g. "joy" and "neutral"
    tokenizer = SimpleTokenizer(vocab_size=10)
    tokenizer.fit(texts)
    dataset = MELDDataset(texts, labels, tokenizer, max_len=5)
    return dataset

def test_dataset_length(sample_dataset):
    """ 
    Checks that the dataset length matches the number of input examples (texts and labels).
    Since we passed in 2 sentences, the dataset should have length 2.
    ✅ Ensures the constructor stored all data correctly.
    """
    assert len(sample_dataset) == 2

def test_dataset_item_is_tuple(sample_dataset):
    """ 
    Takes the first item from the dataset.
    Checks that:
    tokens (the encoded sentence) is a tensor.
    label (the emotion ID) is also a tensor.
    ✅ Ensures MELDDataset is returning PyTorch-friendly objects (not raw lists/ints).
    """
    tokens, label = sample_dataset[0]
    assert isinstance(tokens, torch.Tensor)
    assert isinstance(label, torch.Tensor)

def test_token_shape_and_type(sample_dataset):
    """
    Checks three things about the first item:
    The tokens are stored as integers (torch.long) — required for embedding layers.
    The label is also a torch.long (needed for CrossEntropyLoss).
    The tokens vector has length 5, which matches the max_len we specified.
    ✅ Ensures proper dtype and padding/truncation.
    """
    tokens, label = sample_dataset[0]
    assert tokens.dtype == torch.long
    assert label.dtype == torch.long
    assert tokens.shape[0] == 5   # matches max_len

def test_label_matches_input(sample_dataset):
    """
    Extracts the labels from the first and second items.
    .item() converts a 0D tensor into a plain Python number.
    Checks that the labels match the original list [3, 4] we passed in.
    ✅ Ensures no mix-up between texts and labels.
    """
    _, label0 = sample_dataset[0]
    _, label1 = sample_dataset[1]
    assert label0.item() == 3
    assert label1.item() == 4
