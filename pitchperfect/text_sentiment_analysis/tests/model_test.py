import torch
import pytest
from pitchperfect.text_sentiment_analysis.full_py_files.text_sentiment_pth import BiLSTMEmotionClassifier

@pytest.fixture
def model():
    vocab_size = 100  # dummy vocab size
    return BiLSTMEmotionClassifier(vocab_size=vocab_size)

def test_forward_output_shape(model):
    """
    Forward pass should produce logits of shape (batch_size, num_classes).
    Feeds in 2 sequences of length 10.
    Runs them through the model.
    Asserts the output shape = (2, 7).
    ✅ Checks the model’s output has the right structure: one row per input sequence, one column per emotion class.
    """
    x = torch.randint(0, 100, (2, 10))  # batch_size=2, seq_len=10
    out = model(x)
    assert out.shape == (2, 7)  # 7 = num_classes (emotions)

def test_forward_runs_without_error(model):
    """
    Ensure the forward pass runs without throwing errors. 
    Feeds in 4 sequences of length 15.
    Ensures the model can process them without raising an exception.
    ✅ This is a sanity check: no matter the batch size or sequence length, the forward pass shouldn’t crash.
    """
    x = torch.randint(0, 100, (4, 15))  # batch=4, seq_len=15
    try:
        _ = model(x)
    except Exception as e:
        pytest.fail(f"Forward pass failed with error: {e}")

def test_output_is_tensor(model):
    """
    The output must be a torch.Tensor.
    Feeds in 3 sequences of length 8.
    Asserts the output is a torch.Tensor.
    ✅ Guards against accidental bugs where the model might return a list, NumPy array, or something unexpected.
    """
    x = torch.randint(0, 100, (3, 8))
    out = model(x)
    assert isinstance(out, torch.Tensor)

def test_gradients_flow(model):
    """
    Check that gradients propagate backwards.
    Creates a dummy loss by summing all logits.
    Calls .backward() to trigger backprop.
    Checks at least one parameter got a gradient.
    ✅ Ensures your model isn’t “dead” (e.g., all layers detached or frozen).
    """
    x = torch.randint(0, 100, (2, 12))
    out = model(x)
    loss = out.sum()  # dummy loss
    loss.backward()

    # At least one parameter should have non-None gradients
    has_grad = any(p.grad is not None for p in model.parameters())
    assert has_grad, "No gradients flowed during backward pass!"

def test_padding_embedding_zero(model):
    """
    Padding index (0) should always map to a zero vector in embeddings.
    Passes only padding tokens (ID = 0).
    Checks that their embeddings are all zero vectors.
    ✅ Ensures embedding layer respects padding_idx=0, so padding doesn’t leak information into training.
    """
    pad_token = torch.tensor([[0, 0, 0]])  # batch_size=1, seq_len=3
    embedded = model.embedding(pad_token)
    assert torch.all(embedded == 0), "Padding embeddings should be all zeros"
