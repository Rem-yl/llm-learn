from typing import List, Tuple

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset


class GPTDataset(Dataset):
    def __init__(self, text: str, tokenizer: tiktoken.Encoding, max_len: int, stride: int) -> None:
        """
        Initialize the GPTDataset.

        Example:
        >>> dataset = GPTDataset("Hello, world!I love rem.", tiktoken.get_encoding("gpt2"), 4, 2)
        >>> len(dataset)
        2

        This constructor tokenizes the input text using the provided tokenizer and 
        creates input and target sequences for language modeling. The sequences are 
        created with a specified maximum length and a given stride, allowing for 
        overlapping sequences if desired.

        Args:
            text (str): The input text to be tokenized and converted into sequences.
            tokenizer (tiktoken.Encoding): The tokenizer used to encode the input text.
            max_len (int): The maximum length of each sequence in tokens.
            stride (int): The stride size for creating overlapping sequences.
        """
        super().__init__()
        self.input_ids: List[torch.Tensor] = []
        self.target_ids: List[torch.Tensor] = []
        self.tokenizer = tokenizer

        token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_len, stride):
            input_ids = token_ids[i: i + max_len]
            target_ids = token_ids[i + 1: i + 1 + max_len]
            self.input_ids.append(torch.tensor(input_ids))
            self.target_ids.append(torch.tensor(target_ids))

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.input_ids[index], self.target_ids[index]


def build_gpt_dataloader(text: str, batch_size: int, max_len: int = 256, stride: int = 128, shuffle: bool = True, drop_last: bool = True, num_workers: int = 8) -> DataLoader:
    """
    Builds a PyTorch DataLoader for language modeling tasks with GPT style models.

    Example:
    >>> dataloader = build_gpt_dataloader("Hello, world! I love rem.", batch_size=2, max_len=4, stride=2)
    >>> for input_ids, target_ids in dataloader:
    ...     print(input_ids.shape, target_ids.shape)
    torch.Size([2, 4]) torch.Size([2, 4])

    Args:
        text (str): The input text to be tokenized and converted into sequences.
        batch_size (int): The batch size for the DataLoader.
        max_len (int, optional): The maximum length of each sequence in tokens. Defaults to 256.
        stride (int, optional): The stride size for creating overlapping sequences. Defaults to 128.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        drop_last (bool, optional): Whether to drop the last batch if it is not a full batch. Defaults to True.
        num_workers (int, optional): The number of worker processes to use for data loading. Defaults to 8.

    Returns:
        DataLoader: A PyTorch DataLoader for the provided text.
    """
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDataset(text, tokenizer, max_len, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
