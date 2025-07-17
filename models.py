import torch
import torch.nn as nn

from layers import TransformerBlock


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_len: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the GPT model.

        Args:
            vocab_size (int): The vocabulary size of the input data.
            emb_dim (int): The embedding dimension of the model.
            context_len (int): The maximum context length for the attention mechanism.
            n_heads (int): The number of attention heads.
            n_layers (int): The number of transformer layers.
            dropout (float, optional): The dropout rate for the attention weights and feedforward network. Defaults to 0.1.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_len, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.transformer_blocks = nn.Sequential(
            *[
                TransformerBlock(emb_dim, context_len, n_heads, dropout, False)
                for _ in range(n_layers)
            ]
        )
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.output_layer = nn.Linear(emb_dim, vocab_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs a forward pass through the GPT model.

        Example:
        >>> vocab_size, emb_dim, context_len, n_heads, n_layers = 100, 32, 128, 8, 4
        >>> model = GPTModel(vocab_size, emb_dim, context_len, n_heads, n_layers)
        >>> x = torch.randint(0, 100, (8, 10))
        >>> x.shape
        torch.Size([8, 10])
        >>> logits = model(x)
        >>> logits.shape
        torch.Size([8, 10, 100])

        Args:
            x (torch.Tensor): Input tensor of token indices with shape [batch_size, sequence_length].

        Returns:
            torch.Tensor: Logits tensor with shape [batch_size, sequence_length, vocab_size] after
            passing through token embeddings, positional embeddings, dropout, transformer blocks,
            layer normalization, and the output linear layer.
        """

        x = self.token_embedding(x)
        x = x + self.position_embedding(torch.arange(x.shape[1], device=x.device))
        x = self.dropout_layer(x)
        x = self.transformer_blocks(x)
        x = self.layer_norm(x)
        x = self.output_layer(x)
        return x


if __name__ == "__main__":
    import doctest

    doctest.testmod()
