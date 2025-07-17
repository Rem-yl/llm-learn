import tiktoken
import torch


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
