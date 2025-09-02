# test_text_sentiment_pth.py
import pytest
from pitchperfect.text_sentiment_analysis.full_py_files.text_sentiment_pth import SimpleTokenizer

@pytest.fixture
def tokenizer_and_encoded():
    """
    Creates a SimpleTokenizer.
    Fits it on 4 training sentences (so it learns vocab).
    Encodes the test sentence "i love bananas" into a list of 6 integers (with padding if needed).
    Returns both the fitted tokenizer and the encoded sentence.
    """
    texts = ["i love pears", "pears are great", "i love apples", "apples and pears"]
    tokenizer = SimpleTokenizer(vocab_size=10)
    tokenizer.fit(texts)
    test_sentence = "i love bananas"
    encoded = tokenizer.encode(test_sentence, max_len=6)
    return tokenizer, encoded

def test_vocab_contains_pad_and_oov(tokenizer_and_encoded):
    """
    Checks that <PAD> and <OOV> tokens were correctly initialized in the vocab.
    Ensures special tokens aren’t accidentally dropped.
    """
    tokenizer, _ = tokenizer_and_encoded
    assert "<PAD>" in tokenizer.word2idx
    assert "<OOV>" in tokenizer.word2idx

def test_encode_padding_length(tokenizer_and_encoded):
    """
    Ensures the encode() method always returns exactly max_len tokens.
    Verifies that padding works correctly.
    """
    _, encoded = tokenizer_and_encoded
    assert len(encoded) == 6  # must be padded/truncated to max_len

def test_oov_handling(tokenizer_and_encoded):
    """  
    Confirms that unknown words (like "bananas") are mapped to index 1.
    This is critical so the model knows how to handle unseen vocabulary during inference.
    """
    _, encoded = tokenizer_and_encoded
    assert 1 in encoded  # OOV token id = 1
