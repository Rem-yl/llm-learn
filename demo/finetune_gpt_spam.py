import os
import wandb
from typing import Tuple
import torch
import torch.nn as nn
import hydra
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file
from pathlib import Path
import pandas as pd
from sklearn.metrics import roc_auc_score
from models import GPTModel, download_weights
from datasets import download_and_unzip_spam_data, build_spam_dataloader


def download_and_load_weights(cfg: DictConfig) -> GPTModel:
    download_weights(cfg.model.url, cfg.model.file_name)
    model = GPTModel(vocab_size=cfg.model.vocab_size, emb_dim=cfg.model.emb_dim,
                     context_len=cfg.model.context_len, n_heads=cfg.model.n_heads, n_layers=cfg.model.n_layers, bias=cfg.model.bias)

    state_dict = load_file(cfg.model.file_name)
    model.load_weights(state_dict)

    return model


def split_spam_data(cfg: DictConfig) -> Tuple[Path, Path]:
    download_and_unzip_spam_data(cfg.dataset.url, cfg.dataset.data_root, cfg.dataset.file_name)
    data_root = Path(cfg.dataset.data_root)
    ori_path = data_root / f"{cfg.dataset.file_name}.csv"
    train_path = data_root / f"{cfg.dataset.file_name}_train.csv"
    val_path = data_root / f"{cfg.dataset.file_name}_val.csv"

    df = pd.read_csv(ori_path, sep='\t', header=None, names=["label", "text"])
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    train_df.to_csv(train_path, sep=",", index=False, header=False)
    val_df.to_csv(val_path, sep=",", index=False, header=False)

    return train_path, val_path


@hydra.main(config_path="../conf", config_name="demo_finetune", version_base="1.1")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print("Config:")
    print(OmegaConf.to_yaml(cfg))
    wandb.init(
        project="gpt-spam-finetune",
        name=cfg.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model = download_and_load_weights(cfg)
    train_path, val_path = split_spam_data(cfg)
    train_loader = build_spam_dataloader(train_path, batch_size=cfg.dataset.batch_size, max_len=cfg.dataset.max_len)
    val_loader = build_spam_dataloader(val_path, batch_size=cfg.dataset.batch_size, max_len=cfg.dataset.max_len)

    for param in model.parameters():
        param.requires_grad = False
    model.output_layer = nn.Linear(model.token_embedding.embedding_dim, 1, bias=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.dataset.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    wandb.watch(model, log="all", log_freq=10)

    for epoch in range(cfg.dataset.epochs):
        model.train()
        train_loss = 0.0

        for step, (label, data) in enumerate(train_loader):
            logits = model(data)
            logits = logits[:, -1]
            loss = loss_fn(logits.squeeze(-1), label.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            wandb.log({"train/batch_loss": loss.item(), "train/step": epoch * len(train_loader) + step})

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for step, (label, data) in enumerate(val_loader):
                logits = model(data)
                logits = logits[:, -1]
                loss = loss_fn(logits.squeeze(-1), label.float())
                val_loss += loss.item()
                probs = torch.sigmoid(logits.squeeze(-1))
                all_preds.extend(probs.cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                wandb.log({"val/batch_loss": loss.item(), "val/step": epoch * len(val_loader) + step})

        try:
            auc = roc_auc_score(all_labels, all_preds)
        except ValueError:
            auc = float("nan")  # 有时可能只有一个类，无法计算 AUC

        print(f"Epoch {epoch + 1}, Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss / len(val_loader):.4f}, AUC: {auc:.4f}")
        wandb.log({
            "train/epoch_loss": train_loss / len(train_loader),
            "val/epoch_loss": val_loss / len(val_loader),
            "val/auc": auc,
            "epoch": epoch,
        })

        # ✅ 保存模型并上传到 wandb
        model_path = f"checkpoints/spam_finetune_model_epoch{epoch}.pth"
        torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()
