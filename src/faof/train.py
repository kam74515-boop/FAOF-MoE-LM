from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from contextlib import nullcontext

from .config import TrainConfig
from .data import BinaryTokenDataset
from .model import FAOFModel, ModelConfig
from .tokenizer import CharTokenizer


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"unsupported dtype: {name}")


def autocast_context(device: torch.device, dtype: torch.dtype):
    if dtype == torch.float32:
        return nullcontext()
    if device.type in {"cuda", "mps", "cpu"}:
        return torch.autocast(device_type=device.type, dtype=dtype)
    return nullcontext()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def lr_at_step(step: int, cfg: TrainConfig) -> float:
    if step < cfg.warmup_steps:
        return cfg.learning_rate * (step + 1) / max(1, cfg.warmup_steps)
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return 0.1 * cfg.learning_rate + 0.9 * cfg.learning_rate * 0.5 * (
        1.0 + math.cos(math.pi * progress)
    )


def choose_positions(max_pos: int, count: int, device: torch.device) -> torch.Tensor:
    count = min(count, max_pos)
    return torch.randperm(max_pos, device=device)[:count].sort().values


def make_bridge_targets(
    full_ids: torch.Tensor,
    positions: torch.Tensor,
    offsets: torch.Tensor,
    max_bridge_len: int,
) -> torch.Tensor:
    bsz = full_ids.shape[0]
    targets = torch.full(
        (bsz, len(positions), max_bridge_len),
        -100,
        device=full_ids.device,
        dtype=torch.long,
    )
    for i, (pos, off) in enumerate(zip(positions.tolist(), offsets.tolist(), strict=True)):
        span = full_ids[:, pos + 1 : pos + int(off)]
        targets[:, i, : span.shape[1]] = span
    return targets


def loss_fn(
    model: FAOFModel,
    cfg: TrainConfig,
    full_ids: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    input_ids = full_ids[:, : cfg.block_size]
    order_positions = choose_positions(cfg.block_size, cfg.order_positions, input_ids.device)
    bridge_positions = choose_positions(cfg.block_size, cfg.bridge_positions, input_ids.device)
    bridge_offsets = torch.tensor(
        np.random.choice(cfg.bridge_offsets, size=len(bridge_positions)),
        device=input_ids.device,
        dtype=torch.long,
    )
    out = model(input_ids, order_positions, bridge_positions, bridge_offsets)
    next_logits = out["next_logits"]
    anchor_logits = out["anchor_logits"]

    loss_next = F.cross_entropy(
        next_logits.reshape(-1, next_logits.shape[-1]),
        full_ids[:, 1 : cfg.block_size + 1].reshape(-1),
    )

    loss_anchor = torch.zeros((), device=input_ids.device)
    for k in cfg.offsets:
        logits = anchor_logits[str(k)]
        target = full_ids[:, k : cfg.block_size + k]
        loss_anchor = loss_anchor + F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            target.reshape(-1),
        )
    loss_anchor = loss_anchor / len(cfg.offsets)

    order_logits = out["order_logits"]
    order_target = torch.zeros_like(order_logits)
    for d in range(1, cfg.future_window + 1):
        fut = full_ids[:, order_positions + d]
        order_target.scatter_(2, fut.unsqueeze(-1), 1.0)
    bce = F.binary_cross_entropy_with_logits(order_logits, order_target, reduction="none")
    weights = torch.where(order_target > 0, 1.0, cfg.order_free_neg_weight)
    loss_order = (bce * weights).mean()

    bridge_logits = out["bridge_logits"]
    max_bridge_len = max(cfg.bridge_offsets) - 1
    bridge_target = make_bridge_targets(full_ids, bridge_positions, bridge_offsets, max_bridge_len)
    loss_bridge = F.cross_entropy(
        bridge_logits.reshape(-1, bridge_logits.shape[-1]),
        bridge_target.reshape(-1),
        ignore_index=-100,
    )

    loss = (
        cfg.next_weight * loss_next
        + cfg.anchor_weight * loss_anchor
        + cfg.order_free_weight * loss_order
        + cfg.bridge_weight * loss_bridge
    )
    metrics = {
        "loss": float(loss.detach().cpu()),
        "next": float(loss_next.detach().cpu()),
        "anchor": float(loss_anchor.detach().cpu()),
        "order": float(loss_order.detach().cpu()),
        "bridge": float(loss_bridge.detach().cpu()),
    }
    return loss, metrics


@torch.no_grad()
def evaluate(model: FAOFModel, cfg: TrainConfig, dataset: BinaryTokenDataset, batches: int = 20) -> dict:
    model.eval()
    acc = {"loss": 0.0, "next": 0.0, "anchor": 0.0, "order": 0.0, "bridge": 0.0}
    for _ in range(batches):
        full_ids = dataset.sample_batch(cfg.batch_size)
        _, metrics = loss_fn(model, cfg, full_ids)
        for key in acc:
            acc[key] += metrics[key]
    model.train()
    return {key: val / batches for key, val in acc.items()}


def save_ckpt(path: Path, model: FAOFModel, opt: AdamW, cfg: TrainConfig, step: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "config": cfg.to_dict(), "step": step}
    if cfg.save_optimizer:
        payload["optimizer"] = opt.state_dict()
    torch.save(payload, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = TrainConfig.from_json(args.config)
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    dtype = pick_dtype(cfg.dtype)
    tok = CharTokenizer.load(cfg.tokenizer_path)
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

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
    train_ds = BinaryTokenDataset(cfg.train_bin, cfg.block_size, cfg.extra_tokens, tok.vocab_size, device)
    val_ds = BinaryTokenDataset(cfg.val_bin, cfg.block_size, cfg.extra_tokens, tok.vocab_size, device)
    opt = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    start_step = 0
    if cfg.init_from:
        ckpt = torch.load(cfg.init_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        print(f"initialized model weights from {cfg.init_from}", flush=True)
    if cfg.resume_from:
        ckpt = torch.load(cfg.resume_from, map_location=device)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        start_step = int(ckpt.get("step", 0))
        print(f"resumed from {cfg.resume_from} at step={start_step}", flush=True)

    print(f"device={device} params={sum(p.numel() for p in model.parameters()):,}", flush=True)
    model.train()
    t0 = time.time()
    for step in range(start_step + 1, cfg.max_steps + 1):
        opt.zero_grad(set_to_none=True)
        merged = None
        for _ in range(cfg.grad_accum_steps):
            full_ids = train_ds.sample_batch(cfg.batch_size)
            with autocast_context(device, dtype):
                loss, metrics = loss_fn(model, cfg, full_ids)
            (loss / cfg.grad_accum_steps).backward()
            merged = metrics
        lr = lr_at_step(step, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            print(
                f"step={step} lr={lr:.2e} "
                f"loss={merged['loss']:.4f} next={merged['next']:.4f} "
                f"anchor={merged['anchor']:.4f} order={merged['order']:.4f} "
                f"bridge={merged['bridge']:.4f} tok/s={cfg.batch_size * cfg.block_size * cfg.log_interval / max(dt, 1e-9):.0f}",
                flush=True,
            )
            t0 = time.time()
        if step % cfg.eval_interval == 0:
            val = evaluate(model, cfg, val_ds, batches=5 if cfg.max_steps < 500 else 20)
            print("val " + " ".join(f"{k}={v:.4f}" for k, v in val.items()), flush=True)
        if step % cfg.save_interval == 0 or step == cfg.max_steps:
            save_ckpt(out_dir / "ckpt_last.pt", model, opt, cfg, step)


if __name__ == "__main__":
    main()
