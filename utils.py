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
    torch.Size([1, 4])

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
    '"#$%&'

    Args:
        ids (torch.Tensor): A tensor of shape [1, sequence_length] containing the token ids.
        tokenizer (tiktoken.Encoding): The tokenizer to use for decoding.

    Returns:
        str: The decoded text.
    """
    decoded = tokenizer.decode(ids.squeeze(0).tolist())
    return decoded


def generate_text(model: GPTModel, idx: torch.Tensor, max_tokens: int, context_size: int, use_cache: bool = False) -> torch.Tensor:
    """
    Generate text using a GPT model.

    Example:
    >>> vocab_size, emb_dim, context_len, n_heads, n_layers = 100, 32, 128, 8, 4
    >>> model = GPTModel(vocab_size, emb_dim, context_len, n_heads, n_layers)
    >>> idx = torch.randint(0, 100, (8, 10))
    >>> idx.shape
    torch.Size([8, 10])
    >>> generated_ids = generate_text(model, idx, max_tokens=10, context_size=10, use_cache=True)
    >>> generated_ids.shape
    torch.Size([8, 20])

    Args:
        model (GPTModel): The GPT model to use for generation.
        idx (torch.Tensor): The starting token ids. Shape is [batch_size, sequence_length].
        max_tokens (int): The number of tokens to generate.
        context_size (int): The maximum context size.
        use_cache (bool, optional): Whether to use the cache for the key-value pairs. Defaults to False.

    Returns:
        torch.Tensor: The generated text. Shape is [batch_size, sequence_length + max_tokens].
    """
    model.eval()
    with torch.no_grad():
        if use_cache:
            model._reset_kv_cache()
            logits = model(idx[:, -context_size:], use_cache=True)  # 使用上下文来填充缓存

            for _ in range(max_tokens):
                prob = torch.softmax(logits[:, -1], dim=-1)
                next_id = torch.multinomial(prob, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)
                logits = model(next_id, use_cache=True)  # 每次只需要一个token, 之前的上下文已经保存在缓存中
        else:
            for _ in range(max_tokens):
                logits = model(idx[:, -context_size:], use_cache=False)
                prob = torch.softmax(logits[:, -1], dim=-1)
                next_id = torch.multinomial(prob, num_samples=1)
                idx = torch.cat((idx, next_id), dim=1)

        return idx


if __name__ == "__main__":
    import doctest

    doctest.testmod()
