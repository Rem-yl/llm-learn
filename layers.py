import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_heads: int, context_len: int = 1024, dropout: float = 0.1, bias: bool = False) -> None:
        """
        Initialize the MultiHeadAttention module.
        Example:
        >>> mha = MultiHeadAttention(32, 32, 8)
        >>> x = torch.randn(8, 10, 32)
        >>> out = mha(x)
        >>> out.shape
        torch.Size([8, 10, 32])

        Args:
            in_dim (int): input dimension of the module
            out_dim (int): output dimension of the module
            num_heads (int): number of attention heads
            context_len (int, optional): maximum context length. Defaults to 1024.
            dropout (float, optional): dropout rate for the attention weights. Defaults to 0.1.
            bias (bool, optional): whether to use bias in the linear layers. Defaults to False.

        Raises:
            AssertionError: if out_dim is not divisible by num_heads
        """
        super().__init__()
        assert out_dim % num_heads == 0, "out_dim must be divisible by num_heads"

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.context_len = context_len
        self.dropout = dropout
        self.bias = bias
        self.head_dim = out_dim // num_heads

        self.query_layer = nn.Linear(in_dim, out_dim, bias=bias)
        self.key_layer = nn.Linear(in_dim, out_dim, bias=bias)
        self.value_layer = nn.Linear(in_dim, out_dim, bias=bias)

        self.out_layer = nn.Linear(out_dim, out_dim)
        self.dropout_layer = nn.Dropout(dropout)

        # 不会被当成模型参数, 但是需要保存和加载的张量
        self.register_buffer("mask", torch.triu(torch.ones(context_len, context_len), diagonal=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape  # [b, seq_len, in_dim]

        query = self.query_layer(x)  # [b, seq_len, out_dim]
        key = self.key_layer(x)
        value = self.value_layer(x)

        # [b, seq_len, out_dim] -> [b, num_heads, seq_len, head_dim]
        query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attention_score = query @ key.transpose(2, 3)  # [b, num_heads, seq_len, seq_len]
        attention_score.masked_fill_(self.mask[:seq_len, :seq_len].bool(), -torch.inf)  # 因果注意力
        attention_weight = torch.softmax(attention_score / key.shape[-1] ** 0.5, dim=-1)
        attention_weight = self.dropout_layer(attention_weight)

        out = attention_weight @ value  # [b, num_heads, seq_len, head_dim]
        out = (out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.out_dim))  # [b, seq_len, out_dim]

        return self.out_layer(out)


class GELU(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, in_dim: int) -> None:
        """
        Initialize the FeedForward module.
        Example:
        >>> ffn = FeedForward(32)
        >>> x = torch.randn(8, 10, 32)
        >>> out = ffn(x)
        >>> out.shape
        torch.Size([8, 10, 32])

        Args:
            in_dim (int): input dimension of the module
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, 4 * in_dim), GELU(), nn.Linear(4 * in_dim, in_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, emb_dim: int, context_len: int, num_heads: int, dropout: float, bias: bool
    ) -> None:
        """
        Initialize the TransformerBlock module.

        Example:
        >>> transformer_block = TransformerBlock(32, 128, 8, 0.1, False)

        Args:
            emb_dim (int): The embedding dimension of the model.
            context_len (int): The maximum context length for the attention mechanism.
            num_heads (int): The number of attention heads.
            dropout (float): The dropout rate applied to the attention weights and feedforward network.
            bias (bool): Whether to include bias in linear layers.
        """

        super().__init__()
        self.emb_dim = emb_dim
        self.context_len = context_len
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias

        self.layer_norm1 = nn.LayerNorm(emb_dim)
        self.layer_norm2 = nn.LayerNorm(emb_dim)
        self.mha = MultiHeadAttention(
            emb_dim, emb_dim, num_heads, context_len, dropout, bias
        )
        self.ffn = FeedForward(emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the TransformerBlock module.

        Example:
        >>> transformer_block = TransformerBlock(32, 128, 8, 0.1, False)
        >>> x = torch.randn(8, 10, 32)
        >>> out = transformer_block(x)
        >>> out.shape
        torch.Size([8, 10, 32])

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length, emb_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, emb_dim] after
            passing through layer normalization, multi-head attention, dropout, and feedforward network.
        """
        shortcut = x
        x = self.layer_norm1(x)
        x = self.mha(x)
        x = self.dropout_layer(x)
        x = shortcut + x

        shortcut = x
        x = self.layer_norm2(x)
        x = self.ffn(x)
        x = self.dropout_layer(x)
        x = shortcut + x

        return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
