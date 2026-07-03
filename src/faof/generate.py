from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .model import FAOFModel, ModelConfig
from .tokenizer import CharTokenizer
from .train import pick_device, pick_dtype


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, k)
    cutoff = values[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    cfg = TrainConfig.from_json(args.config)
    device = pick_device(cfg.device)
    dtype = pick_dtype(cfg.dtype)
    tok = CharTokenizer.load(cfg.tokenizer_path)
    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=0.0,
        offsets=cfg.offsets,
        bridge_offsets=cfg.bridge_offsets,
        soft_anchor_top_m=cfg.soft_anchor_top_m,
        activation_checkpointing=False,
    )
    model = FAOFModel(model_cfg).to(device=device, dtype=dtype)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ids = tok.encode(args.prompt, add_eos=False)
    if not ids:
        raise ValueError("prompt produced no tokens")
    x = torch.tensor(ids, device=device, dtype=torch.long)[None, :]

    for _ in range(args.max_new_tokens):
        x_cond = x[:, -cfg.block_size :]
        logits = model(x_cond)["next_logits"][:, -1, :].float()
        if args.temperature <= 0:
            next_id = logits.argmax(dim=-1, keepdim=True)
        else:
            logits = logits / args.temperature
            logits = top_k_filter(logits, args.top_k)
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        x = torch.cat([x, next_id], dim=1)

    print(tok.decode(x[0].tolist()))


if __name__ == "__main__":
    main()

