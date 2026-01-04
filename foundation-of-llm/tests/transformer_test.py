import pytest
import torch
import torch.nn as nn
from transformer import Attention


@pytest.fixture
def emb_dim():
    return 5


@pytest.fixture
def input_tensor(emb_dim):
    # [batch_size, seq_len, emd_dim]
    return torch.randn(4, 3, emb_dim)


class TestTransformer:
    def test_build_attention(self, emb_dim):
        attention_layer = Attention(emb_dim)
        assert attention_layer is not None
        assert isinstance(attention_layer, nn.Module)

    def test_attention_module(self, input_tensor):
        batch_size, seq_len, d_k = input_tensor.shape
        attention_layer = Attention(d_k)

        output, weight = attention_layer(input_tensor)

        assert output.dim() == 3  # [batch_size, seq_len, emb_dim]
        assert output.shape == input_tensor.shape
        assert weight.shape == torch.Size([batch_size, seq_len, seq_len])

        assert torch.allclose(
            torch.sum(weight, dim=-1), torch.ones(batch_size, seq_len)
        )
