import os
import wandb
from typing import Tuple
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf

import json
from pathlib import Path
from models import download_and_load_weights
from datasets import build_instruction_dataloader, InstructionDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def split_instruction_data(cfg: DictConfig) -> Tuple[Path, Path]:
    data_root = Path(cfg.dataset.data_root)
    ori_path = data_root / f"{cfg.dataset.file_name}.json"
    train_path = data_root / f"{cfg.dataset.file_name}_train.json"
    val_path = data_root / f"{cfg.dataset.file_name}_val.json"

    with open(ori_path, "r", encoding="utf-8") as f:
        data = json.load(f)  # ✅ 正确加载整个 JSON 列表

    split_idx = int(len(data) * cfg.dataset.split_rate)
    train_data = data[:split_idx]
    val_data = data[split_idx:]

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)

    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    return train_path, val_path


@hydra.main(config_path="../conf", config_name="demo_instruction", version_base="1.1")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    run = wandb.init(
        project="train-instruction-gpt",
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
    model = download_and_load_weights(cfg)
    train_path, val_path = split_instruction_data(cfg)
    train_loader = build_instruction_dataloader(
        train_path, batch_size=cfg.dataset.batch_size, max_len=cfg.dataset.max_len, device=device)
    val_loader = build_instruction_dataloader(
        val_path, batch_size=cfg.dataset.batch_size, max_len=cfg.dataset.max_len, device=device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.dataset.lr, weight_decay=cfg.dataset.weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    wandb.watch(model, log="all", log_freq=10)
    for epoch in range(cfg.dataset.epochs):
        model.train()
        train_loss = 0.0

        for step, (inputs_tensor, target_tensor) in enumerate(train_loader):
            # logits.shape: [batch_size, seq_len, vocab_size]
            # target_tensor.shape: [batch_size, seq_len]
            logits = model(inputs_tensor)
            loss = loss_fn(logits.flatten(0, 1), target_tensor.flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            wandb.log({"train/batch_loss": loss.item(), "train/step": epoch * len(train_loader) + step})

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for step, (inputs_tensor, target_tensor) in enumerate(train_loader):
                logits = model(inputs_tensor)

                loss = loss_fn(logits.flatten(0, 1), target_tensor.flatten())
                val_loss += loss.item()
                wandb.log({"val/batch_loss": loss.item(), "val/step": epoch * len(val_loader) + step})

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}")
        wandb.log({
            "train/epoch_loss": train_loss / len(train_loader),
            "val/epoch_loss": val_loss / len(val_loader),
            "epoch": epoch,
        })

        val_data = {
            "instruction": "rewrite this sentence as a question.",
            "input": "Your brother went to London last year.",
            "output": "Did your brother go to London last year?"
        }
        text = InstructionDataset.format_input(val_data)
        gened_text = model.gen_text(text, max_tokens=256, context_size=1024, temperature=1.0, top_k=10)
        print(gened_text)

    run.finish()


if __name__ == "__main__":
    main()
