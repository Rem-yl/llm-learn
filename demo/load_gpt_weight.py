import os
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import load_file

from models import GPTModel

url = r"https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors"
output_file = r"model-gpt2.safetensors"


def load_state_dict(output_file) -> Dict[str, torch.Tensor]:
    state_dict = load_file(output_file)
    return state_dict


@hydra.main(config_path="../conf", config_name="default", version_base="1.1")
def main(cfg: DictConfig) -> None:
    os.chdir(hydra.utils.get_original_cwd())
    print("Config:")
    print(OmegaConf.to_yaml(cfg))

    model = GPTModel(vocab_size=cfg.model.vocab_size, emb_dim=cfg.model.emb_dim,
                     context_len=cfg.model.context_len, n_heads=cfg.model.n_heads, n_layers=cfg.model.n_layers, bias=cfg.model.bias)

    state_dict = load_state_dict(output_file)
    model.load_weights(state_dict)
    model.eval()

    txt = "Every effort moves"
    gened_txt = model.gen_text(txt, max_tokens=20, context_size=1024, temperature=1.0, top_k=10)

    print("*" * 50)
    print("Generated Text: \n")
    print(f"{gened_txt}")
    print("*" * 50)


if __name__ == "__main__":
    main()
