from typing import Dict

import tiktoken
import torch
import torch.nn as nn
import os
import urllib.request
from omegaconf import DictConfig

from safetensors.torch import load_file
from layers import TransformerBlock
from utils import assign, ids2text, text2ids


class GPTModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int,
        context_len: int,
        n_heads: int,
        n_layers: int,
        dropout: float = 0.1,
        bias: bool = False
    ) -> None:
        """
        Initialize the GPTModel module.

        Args:
            vocab_size (int): The number of unique tokens in the vocabulary.
            emb_dim (int): The embedding dimension of the model.
            context_len (int): The maximum context length for the attention mechanism.
            n_heads (int): The number of attention heads.
            n_layers (int): The number of TransformerBlock modules.
            dropout (float, optional): The dropout rate applied to the attention weights and feedforward network. Defaults to 0.1.
            bias (bool, optional): Whether to include bias in linear layers. Defaults to False.
        """
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, emb_dim)
        self.position_embedding = nn.Embedding(context_len, emb_dim)
        self.dropout_layer = nn.Dropout(dropout)

        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(emb_dim, context_len, n_heads, dropout,  bias) for _ in range(n_layers)]
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
            pos_ids = torch.arange(self.current_pos, self.current_pos + seq_len,
                                   device=x.device) % self.position_embedding.num_embeddings
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

    def load_weights(self, params: Dict[str, torch.Tensor]):
        """
        #TODO: 加载权重后的模型生成的文体没有语义
        Loads weights from a parameter dictionary into the model.

        Args:
            params (Dict[str, torch.Tensor]): A dictionary mapping parameter names to their weights.
        """

        self.position_embedding.weight = assign(self.position_embedding.weight, params["wpe.weight"])
        self.token_embedding.weight = assign(self.token_embedding.weight, params["wte.weight"])

        for b in range(len(self.transformer_blocks)):
            q_w, k_w, v_w = torch.chunk(
                params[f"h.{b}.attn.c_attn.weight"], 3, dim=-1)
            self.transformer_blocks[b].mha.query_layer.weight = assign(
                self.transformer_blocks[b].mha.query_layer.weight, q_w.T)
            self.transformer_blocks[b].mha.key_layer.weight = assign(
                self.transformer_blocks[b].mha.key_layer.weight, k_w.T)
            self.transformer_blocks[b].mha.value_layer.weight = assign(
                self.transformer_blocks[b].mha.value_layer.weight, v_w.T)

            q_b, k_b, v_b = torch.chunk(
                params[f"h.{b}.attn.c_attn.bias"], 3, axis=-1)
            self.transformer_blocks[b].mha.query_layer.bias = assign(
                self.transformer_blocks[b].mha.query_layer.bias, q_b)
            self.transformer_blocks[b].mha.key_layer.bias = assign(
                self.transformer_blocks[b].mha.key_layer.bias, k_b)
            self.transformer_blocks[b].mha.value_layer.bias = assign(
                self.transformer_blocks[b].mha.value_layer.bias, v_b)

            self.transformer_blocks[b].mha.out_layer.weight = assign(
                self.transformer_blocks[b].mha.out_layer.weight,
                params[f"h.{b}.attn.c_proj.weight"].T)
            self.transformer_blocks[b].mha.out_layer.bias = assign(
                self.transformer_blocks[b].mha.out_layer.bias,
                params[f"h.{b}.attn.c_proj.bias"])

            self.transformer_blocks[b].ffn.layers[0].weight = assign(
                self.transformer_blocks[b].ffn.layers[0].weight,
                params[f"h.{b}.mlp.c_fc.weight"].T)
            self.transformer_blocks[b].ffn.layers[0].bias = assign(
                self.transformer_blocks[b].ffn.layers[0].bias,
                params[f"h.{b}.mlp.c_fc.bias"])
            self.transformer_blocks[b].ffn.layers[2].weight = assign(
                self.transformer_blocks[b].ffn.layers[2].weight,
                params[f"h.{b}.mlp.c_proj.weight"].T)
            self.transformer_blocks[b].ffn.layers[2].bias = assign(
                self.transformer_blocks[b].ffn.layers[2].bias,
                params[f"h.{b}.mlp.c_proj.bias"])

            self.transformer_blocks[b].layer_norm1.weight = assign(
                self.transformer_blocks[b].layer_norm1.weight,
                params[f"h.{b}.ln_1.weight"])
            self.transformer_blocks[b].layer_norm1.bias = assign(
                self.transformer_blocks[b].layer_norm1.bias,
                params[f"h.{b}.ln_1.bias"])
            self.transformer_blocks[b].layer_norm2.weight = assign(
                self.transformer_blocks[b].layer_norm2.weight,
                params[f"h.{b}.ln_2.weight"])
            self.transformer_blocks[b].layer_norm2.bias = assign(
                self.transformer_blocks[b].layer_norm2.bias,
                params[f"h.{b}.ln_2.bias"])

        self.layer_norm.weight = assign(self.layer_norm.weight, params["ln_f.weight"])
        self.layer_norm.bias = assign(self.layer_norm.bias, params["ln_f.bias"])
        self.output_layer.weight = assign(self.output_layer.weight, params["wte.weight"])

    def _generate_text(self, idx: torch.Tensor, max_tokens: int, context_size: int, use_cache: bool = False, temperature=1.0, top_k=None):
        """
        Generate text using a GPT model.

        Args:
            model (GPTModel): The GPT model to use for generation.
            idx (torch.Tensor): The starting token ids. Shape is [batch_size, sequence_length].
            max_tokens (int): The number of tokens to generate.
            context_size (int): The maximum context size.
            use_cache (bool, optional): Whether to use the cache for the key-value pairs. Defaults to False.

        Returns:
            torch.Tensor: The generated text. Shape is [batch_size, sequence_length + max_tokens].
        """
        self.eval()
        with torch.no_grad():
            if use_cache:
                self._reset_kv_cache()
                logits = self.forward(idx[:, -context_size:], use_cache=True)  # 使用上下文来填充缓存

                for _ in range(max_tokens):
                    logits = logits[:, -1]
                    if top_k is not None:
                        top_logits, _ = torch.topk(logits, top_k)
                        min_val = top_logits[:, -1]
                        logits = torch.where(
                            logits < min_val,
                            torch.tensor(float('-inf'), device=logits.device),
                            logits
                        )

                    if temperature > 0.0:
                        logits = logits / temperature

                    prob = torch.softmax(logits, dim=-1)
                    next_id = torch.multinomial(prob, num_samples=1)
                    idx = torch.cat((idx, next_id), dim=1)
                    logits = self.forward(next_id, use_cache=True)  # 每次只需要一个token, 之前的上下文已经保存在缓存中
            else:
                for _ in range(max_tokens):
                    logits = self.forward(idx[:, -context_size:], use_cache=False)
                    logits = logits[:, -1]

                    if top_k is not None:
                        top_logits, _ = torch.topk(logits, top_k)
                        min_val = top_logits[:, -1]
                        logits = torch.where(
                            logits < min_val,
                            torch.tensor(float('-inf'), device=logits.device),
                            logits
                        )

                    if temperature > 0.0:
                        logits = logits / temperature

                    prob = torch.softmax(logits[:, -1], dim=-1)
                    next_id = torch.multinomial(prob, num_samples=1)
                    idx = torch.cat((idx, next_id), dim=1)

            return idx

    def gen_text(self, txt: str, max_tokens: int, context_size: int, temperature=0.0, top_k=None) -> str:
        tokenizer = tiktoken.get_encoding("gpt2")
        idx = text2ids(txt, tokenizer)
        gen_ids = self._generate_text(idx, max_tokens=max_tokens, context_size=context_size,
                                      use_cache=True, temperature=temperature, top_k=top_k)
        gened_txt = ids2text(gen_ids, tokenizer)
        return gened_txt


def download_weights(url: str, file_name: str) -> None:
    """
    Downloads a file from the specified URL and saves it with the given file name.

    Example:
    >>> url = "https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
    >>> file_name = "model.safetensors"
    >>> download_weights(url, file_name)

    Args:
        url (str): The URL from which to download the file.
        file_name (str): The name of the file to save the downloaded content.

    Returns:
        None
    """

    if not os.path.exists(file_name):
        urllib.request.urlretrieve(url, file_name)


def download_and_load_weights(cfg: DictConfig) -> GPTModel:
    """
    Downloads the model weights from the specified URL and loads them into a GPTModel instance.

    Args:
        cfg (DictConfig): Configuration object containing model parameters such as URL, file name, 
                          vocabulary size, embedding dimension, context length, number of heads, 
                          number of layers, and bias setting.

    Returns:
        GPTModel: An instance of the GPTModel with weights loaded from the specified file.
    """

    download_weights(cfg.model.url, cfg.model.file_name)
    model = GPTModel(vocab_size=cfg.model.vocab_size, emb_dim=cfg.model.emb_dim,
                     context_len=cfg.model.context_len, n_heads=cfg.model.n_heads, n_layers=cfg.model.n_layers, bias=cfg.model.bias)

    state_dict = load_file(cfg.model.file_name)
    model.load_weights(state_dict)

    return model


if __name__ == "__main__":
    import doctest

    doctest.testmod()
