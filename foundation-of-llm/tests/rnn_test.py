import pytest
import torch
import torch.nn as nn
from rnn import CustomRNN
from rnn import sentences2id, padding


@pytest.fixture
def vocab():
    return {"<PAD>": 0, "I": 6, "love": 3, "AI": 1, "is": 4, "great": 2, "so": 5}


@pytest.fixture
def sentences():
    return ["I love AI", "AI is so great"]


def test_sentences2id(vocab, sentences):
    word_ids = sentences2id(vocab, sentences)
    for i in range(len(sentences)):
        assert len(word_ids[i]) == len(sentences[i].split())


def test_padding(vocab, sentences):
    default_idx = vocab["<PAD>"]

    word_ids = sentences2id(vocab, sentences)
    input_tensor = padding(word_ids, default_idx)
    max_len = max([len(word_id) for word_id in word_ids])

    assert input_tensor.dim() == 2
    assert input_tensor.shape[0] == len(word_ids)
    assert input_tensor.shape[1] == max_len


class TestCustomRNN:
    def test_build_rnn(self, vocab):
        model = CustomRNN(len(vocab), 5, 16)
        assert isinstance(model, nn.Module)

    def test_embedding(self, vocab, sentences):
        model = CustomRNN(len(vocab), 5, 16)
        input_tensor = padding(sentences2id(vocab, sentences))

        embedding_tensor = model.embedding[input_tensor]

        assert embedding_tensor.dim() == 3
        assert embedding_tensor.shape[:2] == input_tensor.shape
        assert embedding_tensor.shape[2] == model.embed_size
