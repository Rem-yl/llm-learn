import os
import urllib.request
from pathlib import Path
from typing import Union

import hydra
import tiktoken
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf

import wandb
from datasets import build_gpt_dataloader
from models import GPTModel
from utils import generate_text, ids2text, text2ids

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = Path(r"data/the-verdict.txt")
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"


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


@hydra.main(config_path="../conf", config_name="demo_gpt", version_base="1.1")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print("Config:")
    print(OmegaConf.to_yaml(cfg))

    wandb.init(project=cfg.wandb.project, name=cfg.wandb.name)
    download_data(cfg.dataset.url)
    train_loader, val_loader = get_train_val_dataloader(cfg.dataset.file_path)
    model = GPTModel(vocab_size=cfg.model.vocab_size, emb_dim=cfg.model.emb_dim,
                     context_len=cfg.model.context_len, n_heads=cfg.model.n_heads, n_layers=cfg.model.n_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    loss_fn = nn.CrossEntropyLoss()

    gen_txt = "Hello, i love"
    tokenizer = tiktoken.get_encoding("gpt2")
    epochs = cfg.dataset.epochs
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        val_loss = 0.0
        for input_ids, target_ids in train_loader:
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            logits = model(input_ids)
            loss = loss_fn(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        with torch.no_grad():
            for input_ids, target_ids in val_loader:
                input_ids = input_ids.to(device)
                target_ids = target_ids.to(device)

                logits = model(input_ids)
                loss = loss_fn(logits.view(-1, logits.shape[-1]), target_ids.view(-1))
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss = val_loss / len(val_loader.dataset)
        print(
            f"\nEpoch: {epoch + 1}/{epochs} | Train Loss: {avg_train_loss:.5f} | Val Loss: {avg_val_loss:.5f}\n")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss
        })

        idx = text2ids(gen_txt, tokenizer)
        gen_ids = generate_text(model, idx, max_tokens=10, context_size=128, use_cache=True, temperature=0.7, top_k=10)
        gened_txt = ids2text(gen_ids, tokenizer)
        print("*" * 50)
        print("Generated Text: \n")
        print(f"{gened_txt}")
        print("*" * 50)


if __name__ == "__main__":
    main()
