from typing import List, Tuple, Dict

import tiktoken
import torch
from torch.utils.data import DataLoader, Dataset
import os
from pathlib import Path
import shutil
import zipfile
import urllib.request
import pandas as pd
from log import logger
import json
from functools import partial


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


def download_and_unzip_spam_data(url: str, data_root: str | Path, file_name: str) -> None:
    """
    Downloads a zip file containing a text dataset, unzips it, and renames the extracted text file to a CSV file.
    Example:
    >>> url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    >>> data_root = "data"
    >>> file_name = "sms_spam_collection"
    >>> download_and_unzip_spam_data(url, data_root, file_name)
    CSV file already exists at data/sms_spam_collection.csv

    Args:
        url (str): The URL of the zip file containing the text dataset.
        data_root (str | Path): The directory where the zip file should be saved.
        file_name (str): The base name of the saved CSV file.

    Returns:
        None
    """
    if isinstance(data_root, str):
        data_root = Path(data_root)

    if not data_root.exists():
        data_root.mkdir(parents=True)

    zip_path = data_root / f"{file_name}.zip"
    csv_path = data_root / f"{file_name}.csv"
    ori_path = data_root / f"{file_name}" / "SMSSpamCollection"

    if csv_path.exists():
        print(f"CSV file already exists at {csv_path}")
        return

    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root/file_name)

    os.rename(ori_path, csv_path)
    print(f"File downloaded and saved as {csv_path}")
    zip_path.unlink()
    shutil.rmtree(ori_path.parent)


class SpamDataset(Dataset):
    def __init__(self, csv_path: str | Path, max_len: int = 1024, tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")) -> None:
        """
        Initialize the SpamDataset.

        Args:
            csv_path (str | Path): The path to the CSV file containing the text dataset.
            max_len (int, optional): The maximum length of each sequence in tokens. Defaults to 1024.
            tokenizer (tiktoken.Encoding, optional): The tokenizer to use for encoding the text. Defaults to gpt2.

        Returns:
            None
        """
        if isinstance(csv_path, str):
            csv_path = Path(csv_path)

        self.csv_path = csv_path
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.label = []
        self.data = []
        self._load_dataset()

    def _load_dataset(self):
        df = pd.read_csv(self.csv_path, sep=",", header=None, names=["label", "text"])
        pad_id = self.tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
        label_map = {"ham": 0, "spam": 1}

        for _, row in df.iterrows():
            label = label_map[row["label"]]
            text = row["text"]
            token_ids = self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

            # padding
            if len(token_ids) > self.max_len:
                token_ids = token_ids[:self.max_len]
            else:
                token_ids += [pad_id] * (self.max_len - len(token_ids))

            self.data.append(torch.tensor(token_ids))
            self.label.append(torch.tensor(label))

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx: int):
        label = self.label[idx]
        ids = self.data[idx]

        return label, ids


def build_spam_dataloader(csv_path: str | Path, batch_size: int = 8, max_len: int = 1024, tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")) -> DataLoader:
    """
    Builds a PyTorch DataLoader for the Spam dataset.

    Args:
        csv_path (str | Path): The path to the CSV file containing the text dataset.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 8.
        max_len (int, optional): The maximum length of each sequence in tokens. Defaults to 1024.
        tokenizer (tiktoken.Encoding, optional): The tokenizer to use for encoding the text. Defaults to gpt2.

    Returns:
        DataLoader: A PyTorch DataLoader for the Spam dataset.
    """
    dataset = SpamDataset(csv_path, max_len, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def download_instruction_data(url: str, data_root: str | Path, file_name: str = "instruction.json") -> None:
    """
    Downloads a file from the specified URL and saves it with the given file name.

    Example:
    >>> url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    >>> data_root = "data"
    >>> file_name = "instruction.json"
    >>> download_instruction_data(url, data_root, file_name)

    Args:
        url (str): The URL from which to download the file.
        data_root (str | Path): The directory where the file should be saved.
        file_name (str, optional): The name of the file to save the downloaded content. Defaults to "instruction.json".

    Returns:
        None
    """
    if isinstance(data_root, str):
        data_root = Path(data_root)

    if not data_root.exists():
        data_root.mkdir(parents=True)

    ori_path = data_root / file_name

    if ori_path.exists():
        logger.info(f"File already exists at {ori_path}")
        return

    with urllib.request.urlopen(url) as response:
        text_data = response.read().decode("utf-8")

    with open(ori_path, "w", encoding="utf-8") as file:
        file.write(text_data)

    logger.info(f"File downloaded and saved as {ori_path}")


class InstructionDataset(Dataset):
    def __init__(self, path: str | Path, tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2")) -> None:
        super().__init__()

        if isinstance(path, str):
            path = Path(path)

        self.path = path
        self.tokenizer = tokenizer
        self.data: List[Dict[str, str]] = []
        self.encoded_texts: List[List[int]] = []
        self._load_dataset()

    @staticmethod
    def format_input(entry: Dict[str, str]) -> str:
        instruction_text = (
            f"Below is an instruction that describes a task. "
            f"Write a response that appropriately completes the request."
            f"\n\n### Instruction:\n{entry['instruction']}"
        )

        input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

        return instruction_text + input_text

    def _load_dataset(self) -> None:
        with open(self.path, "r", encoding="utf-8") as file:
            self.data = json.load(file)

        for entry in self.data:
            instruction_text = self.format_input(entry)
            response_text = f"\n\n### Response:\n{entry['output']}"
            full_text = instruction_text + response_text
            self.encoded_texts.append(
                self.tokenizer.encode(full_text)
            )

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def instruction_collate_fn(batch, pad_token_id: int = 50256, ignore_index: int = -100, max_len: int = 1024, device="cpu"):
    """
    A collate function for the instruction dataset.
    Example:
    >>> batch = [[0, 1, 2, 3, 4], [5, 6], [7, 8, 9]]
    >>> inputs, targets = instruction_collate_fn(batch)
    >>> inputs
    tensor([[    0,     1,     2,     3,     4],
            [    5,     6, 50256, 50256, 50256],
            [    7,     8,     9, 50256, 50256]])
    >>> targets
    tensor([[    1,     2,     3,     4, 50256],
            [    6, 50256,  -100,  -100,  -100],
            [    8,     9, 50256,  -100,  -100]])

    Args:
        batch: A list of instruction examples.
        pad_token_id: The token id to use for padding.
        ignore_index: The index to use for padding in the target tensor.
        max_len: The maximum length of the input and target tensors.
        device: The device to use for the tensors.

    Returns:
        A tuple of two tensors, the input tensor and the target tensor.
    """
    batch_max_length = max(len(item)+1 for item in batch)

    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = (
            new_item + [pad_token_id] *
            (batch_max_length - len(new_item))
        )
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index  # 计算交叉熵损失时忽略

        if max_len:
            inputs = inputs[:max_len]
            targets = targets[:max_len]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)
    return inputs_tensor, targets_tensor


def build_instruction_dataloader(
    path: str | Path,
    batch_size: int = 8,
    ignore_index: int = -100,
    max_len: int = 1024,
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding("gpt2"),
    device="cpu",
) -> DataLoader:
    """
    Builds a PyTorch DataLoader for the instruction dataset.

    Args:
        path (str | Path): The path to the JSON file containing the instruction dataset.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 8.
        ignore_index (int, optional): The index to use for padding in the target tensor. Defaults to -100.
        max_len (int, optional): The maximum length of the input and target tensors. Defaults to 1024.
        tokenizer (tiktoken.Encoding, optional): The tokenizer to use for encoding the text. Defaults to gpt2.
        device (str, optional): The device to use for the tensors. Defaults to "cpu".

    Returns:
        DataLoader: A PyTorch DataLoader for the instruction dataset.
    """
    dataset = InstructionDataset(path, tokenizer)
    pad_token_id = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})[0]
    collate_fn = partial(
        instruction_collate_fn,
        pad_token_id=pad_token_id,
        ignore_index=ignore_index,
        max_len=max_len,
        device=device,
    )

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
