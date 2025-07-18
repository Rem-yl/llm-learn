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

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, context_len, n_heads, dropout, False) for _ in range(n_layers)]
        )
        self.layer_norm = nn.LayerNorm(emb_dim)
        self.output_layer = nn.Linear(emb_dim, vocab_size, bias=False)
        self.current_pos = 0

    def forward(self, x: torch.Tensor, use_cache=False) -> torch.Tensor:
        """
        Perform a forward pass through the GPT model.

        Example:
        >>> model = GPTModel(100, 32, 128, 8, 4)
        >>> x = torch.randint(0, 100, (8, 10))
        >>> x.shape
        torch.Size([8, 10])
        >>> out = model(x)
        >>> out.shape
        torch.Size([8, 10, 100])

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, sequence_length].
            use_cache (bool, optional): Whether to use the cache for the key-value pairs. Defaults to False.

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, sequence_length, vocab_size].
        """
        _, seq_len = x.shape
        x = self.token_embedding(x)
        if use_cache:
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len, device=x.device)
            self.current_pos += seq_len
        else:
            pos_ids = torch.arange(0, seq_len, device=x.device)

        pos_embedding = self.position_embedding(pos_ids).unsqueeze(0)
        x += pos_embedding
        x = self.dropout_layer(x)
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x, use_cache=use_cache)
        x = self.layer_norm(x)
        x = self.output_layer(x)
        return x

    def _reset_kv_cache(self):
        for transformer_block in self.transformer_blocks:
            transformer_block.mha._reset_cache()

        self.current_pos = 0


if __name__ == "__main__":
    import doctest

    doctest.testmod()
