from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from .config import TrainConfig
from .data import BinaryTokenDataset
from .model import FAOFModel, ModelConfig
from .tokenizer import CharTokenizer
from .train import autocast_context, make_bridge_targets, pick_device, pick_dtype, choose_positions


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--batches", type=int, default=20)
    args = parser.parse_args()

    cfg = TrainConfig.from_json(args.config)
    device = pick_device(cfg.device)
    dtype = pick_dtype(cfg.dtype)
    tok = CharTokenizer.load(cfg.tokenizer_path)
    mcfg = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=cfg.dropout,
        offsets=cfg.offsets,
        bridge_offsets=cfg.bridge_offsets,
        soft_anchor_top_m=cfg.soft_anchor_top_m,
        activation_checkpointing=cfg.activation_checkpointing,
    )
    model = FAOFModel(mcfg).to(device=device, dtype=dtype)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    ds = BinaryTokenDataset(cfg.val_bin, cfg.block_size, cfg.extra_tokens, tok.vocab_size, device)

    sums = {
        "next_loss": 0.0,
        "order_recall": 0.0,
        "order_precision": 0.0,
        "bridge_acc": 0.0,
    }
    anchor_top1 = {k: 0.0 for k in cfg.offsets}
    anchor_top5 = {k: 0.0 for k in cfg.offsets}

    for _ in range(args.batches):
        full_ids = ds.sample_batch(cfg.batch_size)
        input_ids = full_ids[:, : cfg.block_size]
        order_positions = choose_positions(cfg.block_size, cfg.order_positions, device)
        bridge_positions = choose_positions(cfg.block_size, cfg.bridge_positions, device)
        bridge_offsets = torch.tensor(
            np.random.choice(cfg.bridge_offsets, size=len(bridge_positions)),
            device=device,
            dtype=torch.long,
        )
        with autocast_context(device, dtype):
            out = model(input_ids, order_positions, bridge_positions, bridge_offsets)
        next_logits = out["next_logits"]
        sums["next_loss"] += float(
            F.cross_entropy(
                next_logits.reshape(-1, next_logits.shape[-1]),
                full_ids[:, 1 : cfg.block_size + 1].reshape(-1),
            ).cpu()
        )

        for k in cfg.offsets:
            logits = out["anchor_logits"][str(k)]
            target = full_ids[:, k : cfg.block_size + k]
            pred1 = logits.argmax(dim=-1)
            top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
            anchor_top1[k] += float((pred1 == target).float().mean().cpu())
            anchor_top5[k] += float((top5 == target.unsqueeze(-1)).any(dim=-1).float().mean().cpu())

        order_logits = out["order_logits"]
        topn = min(cfg.future_window, order_logits.shape[-1])
        pred_set = order_logits.topk(topn, dim=-1).indices
        target = torch.zeros_like(order_logits, dtype=torch.bool)
        for d in range(1, cfg.future_window + 1):
            fut = full_ids[:, order_positions + d]
            target.scatter_(2, fut.unsqueeze(-1), True)
        hits = target.gather(2, pred_set).sum(dim=-1).float()
        target_count = target.sum(dim=-1).clamp_min(1).float()
        sums["order_recall"] += float((hits / target_count).mean().cpu())
        sums["order_precision"] += float((hits / topn).mean().cpu())

        bridge_logits = out["bridge_logits"]
        bridge_target = make_bridge_targets(
            full_ids, bridge_positions, bridge_offsets, max(cfg.bridge_offsets) - 1
        )
        mask = bridge_target.ne(-100)
        bridge_pred = bridge_logits.argmax(dim=-1)
        sums["bridge_acc"] += float((bridge_pred[mask] == bridge_target[mask]).float().mean().cpu())

    denom = float(args.batches)
    print(f"next_loss={sums['next_loss'] / denom:.4f}")
    for k in cfg.offsets:
        print(f"anchor_top1@{k}={anchor_top1[k] / denom:.4f} anchor_top5@{k}={anchor_top5[k] / denom:.4f}")
    print(f"order_recall={sums['order_recall'] / denom:.4f}")
    print(f"order_precision={sums['order_precision'] / denom:.4f}")
    print(f"bridge_acc={sums['bridge_acc'] / denom:.4f}")


if __name__ == "__main__":
    main()
