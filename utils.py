import tiktoken
import torch

from models import GPTModel


def text2ids(text: str, tokenizer: tiktoken.Encoding) -> torch.Tensor:
    """
    Convert a given text into a tensor of token ids.
    Example:
    >>> text = "Hello, world!"
    >>> tokenizer = tiktoken.get_encoding("gpt2")
    >>> ids = text2ids(text, tokenizer)
    >>> ids.shape
    torch.Size([1, 5])

    Args:
        text (str): The text to convert.
        tokenizer (tiktoken.Encoding): The tokenizer to use for encoding.

    Returns:
        torch.Tensor: A tensor of shape [1, sequence_length] containing the token ids.
    """
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # add batch dimension
    return encoded_tensor


def ids2text(ids: torch.Tensor, tokenizer: tiktoken.Encoding) -> str:
    """
    Convert a tensor of token ids into a string of text.
    Example:
    >>> ids = torch.tensor([[1, 2, 3, 4, 5]])
    >>> tokenizer = tiktoken.get_encoding("gpt2")
    >>> text = ids2text(ids, tokenizer)
    >>> text
    'Hello'

    Args:
        ids (torch.Tensor): A tensor of shape [1, sequence_length] containing the token ids.
        tokenizer (tiktoken.Encoding): The tokenizer to use for decoding.

    Returns:
        str: The decoded text.
    """
    decoded = tokenizer.decode(ids.squeeze(0).tolist())
    return decoded


def generate_text(model: GPTModel, idx: torch.Tensor, max_tokens: int, context_size: int) -> torch.Tensor:
    """
    Generate text using the given model.

    This function takes an input tensor idx, runs it through the model, and generates
    the next token based on the output probabilities. This is repeated until max_tokens
    tokens have been generated.

    Example:
    >>> vocab_size, emb_dim, context_len, n_heads, n_layers = 100, 32, 128, 8, 4
    >>> model = GPTModel(vocab_size, emb_dim, context_len, n_heads, n_layers)
    >>> idx = torch.randint(0, 100, (8, 10))
    >>> idx.shape
    torch.Size([8, 10])
    >>> generated_text = generate_text(model, idx, 10, 10)
    >>> generated_text.shape
    torch.Size([8, 20])

    Args:
        model (GPTModel): The model to use for generation.
        idx (torch.Tensor): The input tensor of shape [batch_size, sequence_length].
        max_tokens (int): The maximum number of tokens to generate.
        context_size (int): The number of tokens to use as context when generating the next token.

    Returns:
        torch.Tensor: The generated text as a tensor of shape [batch_size, sequence_length].
    """
    for _ in range(max_tokens):
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)
            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

    return idx
