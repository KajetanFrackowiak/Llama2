import pytest
from unittest.mock import MagicMock
import torch
import sentencepiece as spm
from model import ModelArgs, Transformer
from inference import LLaMA

@pytest.fixture
def mock_model():
    return MagicMock(spec=Transformer)

@pytest.fixture
def mock_tokenizer():
    return MagicMock(spec=spm.SentencePieceProcessor)

@pytest.fixture
def model_args():
    return ModelArgs(
        max_seq_len=1024,
        max_batch_size=1,
        device='cpu',
        dim=512,
        n_layers=6,
        n_heads=8
    )

def test_build(mock_model, mock_tokenizer, model_args):
    llama = LLaMA(mock_model, mock_tokenizer, model_args)
    assert isinstance(llama, LLaMA)
    
def test_text_completion(mock_model, mock_tokenizer, model_args):
    # Mocking token encoding and decoding
    mock_tokenizer.encode.return_value = [1, 2, 3]
    mock_tokenizer.decode.return_value = ["Sample text"] 

    # Mocking model text completion
    mock_model.forward.return_value = torch.tensor([[0.1, 0.2, 0.3]])

    llama = LLaMA(mock_model, mock_tokenizer, model_args)
    out_tokens, out_text = llama.text_completion(['prompt'])

    # Assert that token encoding and decoding methods are called
    mock_tokenizer.encode.assert_called_once_with('prompt', out_type=int, add_bos=True, add_eos=False)
    mock_tokenizer.tokenizer.decode.assert_called_once_with([1, 2, 3])

    # Assert that model forward method is called
    mock_model.forward.assert_called_once()
    
    # Assert that output tokens and text have correct format
    assert isinstance(out_tokens, list)
    assert isinstance(out_text, list)
    assert len(out_tokens) == 1
    assert len(out_text) == 1
    assert isinstance(out_tokens[0], list)
    assert isinstance(out_text[0], str)