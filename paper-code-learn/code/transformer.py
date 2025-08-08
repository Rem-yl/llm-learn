import torch
import torch.nn as nn
from typing import Tuple, List, Union
import math
import tiktoken
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import urllib.request

import torch.nn.functional as F
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = Path(r"data/the-verdict.txt")
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"


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


def download_data(url: str) -> None:
    if not file_path.exists():
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)


def get_train_val_dataloader(file_path: Union[Path, str], train_ratio=0.9):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader = build_gpt_dataloader(train_data, batch_size=2, max_len=256, stride=128, num_workers=0)
    val_loader = build_gpt_dataloader(val_data, batch_size=2, max_len=256, stride=128, num_workers=0)

    return train_loader, val_loader


class Transformer(nn.Module):
    def __init__(self,
                 enc_voc_size: int,
                 dec_voc_size: int,
                 d_model: int,
                 n_head: int,
                 ffn_hidden: int,
                 n_layers: int,
                 drop_prob: float = 0.1,
                 max_len: int = 1024,
                 device="cpu"):
        super().__init__()

        self.enc_voc_size = enc_voc_size
        self.dec_voc_size = dec_voc_size
        self.d_model = d_model
        self.n_head = n_head
        self.ffn_hidden = ffn_hidden
        self.n_layers = n_layers
        self.drop_prob = drop_prob
        self.max_len = max_len
        self.device = device
