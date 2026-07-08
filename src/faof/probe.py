"""Frozen-backbone probe evaluation.

The ablation success criterion "D's anchor accuracy beats A's" is invalid as
written: in the A/B/C configs the anchor/order/bridge heads receive zero loss
weight, so they stay at random init and any trained head trivially beats them.

This probe answers the question that actually matters: does the FAOF-trained
*backbone representation* carry more decodable future information? For each
checkpoint we freeze the entire model, attach a *fresh* linear probe head for
each auxiliary task, train only the probe (identical budget for every
checkpoint), and evaluate on the validation set. Differences then come from the
frozen features alone, so A becomes a real baseline instead of a random head.

Scope: anchor and order-free probes (clean linear read-outs of the hidden
state). Next-token loss is already a valid comparison (the next head is trained
with weight 1.0 in every config) and is reported for context. Bridge is a
conditional generator entangled with the anchor soft-embeddings; a fair
frozen-backbone probe for it is a separate design and is intentionally omitted.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW

from .config import TrainConfig
from .data import BinaryTokenDataset
from .model import FAOFModel, ModelConfig
from .tokenizer import CharTokenizer
from .train import autocast_context, choose_positions, pick_device, pick_dtype


def build_model(cfg: TrainConfig, vocab_size: int, device, dtype) -> FAOFModel:
    mcfg = ModelConfig(
        vocab_size=vocab_size,
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
    return FAOFModel(mcfg).to(device=device, dtype=dtype)


class ProbeHeads(nn.Module):
    """Fresh, independently-initialised read-out heads trained on frozen features."""

    def __init__(self, cfg: TrainConfig, vocab_size: int):
        super().__init__()
        self.anchor = nn.ModuleDict(
            {str(k): nn.Linear(cfg.n_embd, vocab_size, bias=False) for k in cfg.offsets}
        )
        self.order = nn.Linear(cfg.n_embd, vocab_size, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)


@torch.no_grad()
def backbone_hidden(model: FAOFModel, input_ids: torch.Tensor) -> torch.Tensor:
    """Hidden state from the frozen backbone (heads discarded)."""
    return model(input_ids)["hidden"].detach()


def anchor_loss(probe: ProbeHeads, cfg: TrainConfig, h: torch.Tensor, full_ids: torch.Tensor) -> torch.Tensor:
    loss = torch.zeros((), device=h.device)
    for k in cfg.offsets:
        logits = probe.anchor[str(k)](h)
        target = full_ids[:, k : cfg.block_size + k]
        loss = loss + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))
    return loss / len(cfg.offsets)


def order_loss(
    probe: ProbeHeads,
    cfg: TrainConfig,
    h: torch.Tensor,
    full_ids: torch.Tensor,
    order_positions: torch.Tensor,
) -> torch.Tensor:
    order_logits = probe.order(h[:, order_positions])
    order_target = torch.zeros_like(order_logits)
    for d in range(1, cfg.future_window + 1):
        fut = full_ids[:, order_positions + d]
        order_target.scatter_(2, fut.unsqueeze(-1), 1.0)
    bce = F.binary_cross_entropy_with_logits(order_logits, order_target, reduction="none")
    weights = torch.where(order_target > 0, 1.0, cfg.order_free_neg_weight)
    return (bce * weights).mean()


def train_probe(
    model: FAOFModel,
    probe: ProbeHeads,
    cfg: TrainConfig,
    train_ds: BinaryTokenDataset,
    device: torch.device,
    dtype: torch.dtype,
    steps: int,
    lr: float,
) -> None:
    opt = AdamW(probe.parameters(), lr=lr, weight_decay=0.0)
    probe.train()
    for step in range(1, steps + 1):
        full_ids = train_ds.sample_batch(cfg.batch_size)
        input_ids = full_ids[:, : cfg.block_size]
        order_positions = choose_positions(cfg.block_size, cfg.order_positions, device)
        with autocast_context(device, dtype):
            h = backbone_hidden(model, input_ids)
            loss = anchor_loss(probe, cfg, h, full_ids) + order_loss(
                probe, cfg, h, full_ids, order_positions
            )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0:
            print(f"  probe step {step}/{steps} loss={float(loss.detach().cpu()):.4f}", flush=True)


@torch.no_grad()
def eval_probe(
    model: FAOFModel,
    probe: ProbeHeads,
    cfg: TrainConfig,
    val_ds: BinaryTokenDataset,
    device: torch.device,
    dtype: torch.dtype,
    batches: int,
) -> dict:
    probe.eval()
    anchor_top5 = {k: 0.0 for k in cfg.offsets}
    anchor_top1 = {k: 0.0 for k in cfg.offsets}
    order_recall = 0.0
    next_loss = 0.0
    for _ in range(batches):
        full_ids = val_ds.sample_batch(cfg.batch_size)
        input_ids = full_ids[:, : cfg.block_size]
        order_positions = choose_positions(cfg.block_size, cfg.order_positions, device)
        with autocast_context(device, dtype):
            out = model(input_ids)
            h = out["hidden"]
            next_logits = out["next_logits"]
            next_loss += float(
                F.cross_entropy(
                    next_logits.reshape(-1, next_logits.shape[-1]),
                    full_ids[:, 1 : cfg.block_size + 1].reshape(-1),
                ).cpu()
            )
            for k in cfg.offsets:
                logits = probe.anchor[str(k)](h)
                target = full_ids[:, k : cfg.block_size + k]
                pred1 = logits.argmax(dim=-1)
                top5 = logits.topk(min(5, logits.shape[-1]), dim=-1).indices
                anchor_top1[k] += float((pred1 == target).float().mean().cpu())
                anchor_top5[k] += float((top5 == target.unsqueeze(-1)).any(dim=-1).float().mean().cpu())

            order_logits = probe.order(h[:, order_positions])
            topn = min(cfg.future_window, order_logits.shape[-1])
            pred_set = order_logits.topk(topn, dim=-1).indices
            target = torch.zeros_like(order_logits, dtype=torch.bool)
            for d in range(1, cfg.future_window + 1):
                fut = full_ids[:, order_positions + d]
                target.scatter_(2, fut.unsqueeze(-1), True)
            hits = target.gather(2, pred_set).sum(dim=-1).float()
            order_recall += float((hits / target.sum(dim=-1).clamp_min(1).float()).mean().cpu())

    n = float(batches)
    return {
        "next_loss": next_loss / n,
        "anchor_top1": {k: v / n for k, v in anchor_top1.items()},
        "anchor_top5": {k: v / n for k, v in anchor_top5.items()},
        "order_recall": order_recall / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--probe-steps", type=int, default=300)
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = TrainConfig.from_json(args.config)
    device = pick_device(cfg.device)
    dtype = pick_dtype(cfg.dtype)
    tok = CharTokenizer.load(cfg.tokenizer_path)

    model = build_model(cfg, tok.vocab_size, device, dtype)
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    probe = ProbeHeads(cfg, tok.vocab_size).to(device=device, dtype=dtype)
    train_ds = BinaryTokenDataset(cfg.train_bin, cfg.block_size, cfg.extra_tokens, tok.vocab_size, device)
    val_ds = BinaryTokenDataset(cfg.val_bin, cfg.block_size, cfg.extra_tokens, tok.vocab_size, device)

    print(f"probing {args.ckpt} (frozen backbone, fresh probe, {args.probe_steps} steps)", flush=True)
    train_probe(model, probe, cfg, train_ds, device, dtype, args.probe_steps, args.probe_lr)
    m = eval_probe(model, probe, cfg, val_ds, device, dtype, args.eval_batches)

    print(f"next_loss={m['next_loss']:.4f}  (valid as-is; next head trained in every config)")
    for k in cfg.offsets:
        print(f"probe_anchor_top1@{k}={m['anchor_top1'][k]:.4f} probe_anchor_top5@{k}={m['anchor_top5'][k]:.4f}")
    print(f"probe_order_recall={m['order_recall']:.4f}")


if __name__ == "__main__":
    main()
