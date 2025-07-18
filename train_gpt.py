import urllib.request
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn

import wandb
from datasets import build_gpt_dataloader
from models import GPTModel
from utils import generate_text, ids2text, text2ids

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

file_path = Path(r"data/the-verdict.txt")
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"


def download_data() -> None:
    if not file_path.exists():
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)


def get_train_val_dataloader(file_path: Path, train_ratio=0.9):
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    train_loader = build_gpt_dataloader(train_data, batch_size=2, max_len=256, stride=128, num_workers=0)
    val_loader = build_gpt_dataloader(val_data, batch_size=2, max_len=256, stride=128, num_workers=0)

    return train_loader, val_loader


def main():
    wandb.init(project="GPTModel Train", name="Demo")

    download_data()
    train_loader, val_loader = get_train_val_dataloader(file_path)

    model = GPTModel(vocab_size=50257, emb_dim=32, context_len=1024, n_heads=8, n_layers=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    gen_txt = "Hello, i love"
    tokenizer = tiktoken.get_encoding("gpt2")
    epochs = 50
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
        gen_ids = generate_text(model, idx, max_tokens=10, context_size=128, use_cache=True)
        gened_txt = ids2text(gen_ids, tokenizer)
        print("*" * 50)
        print("Generated Text: \n")
        print(f"{gened_txt}")
        print("*" * 50)


if __name__ == "__main__":
    main()
