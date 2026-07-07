"""Multi-seed, probe-corrected FAOF ablation.

For every (config variant, seed) it trains a backbone from scratch, then runs
the frozen-backbone probe (fresh probe head, identical procedure for all
backbones) and records the metrics. Results are aggregated to mean +/- std
across seeds so the A/B/C/D comparison has error bars.

Why this exists: the repo's original success criterion ("D's anchor top5 beats
A's") is invalid, because in A/B/C the anchor/order/bridge heads get zero loss
weight and stay at random init, so any trained head trivially wins. The probe
freezes each backbone and trains a fresh, identical read-out head, isolating
representation quality. See src/faof/probe.py.

Writes incremental rows to <out>/rows.jsonl and a final aggregate to
<out>/aggregate.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from faof.config import TrainConfig
from faof.data import BinaryTokenDataset
from faof.probe import ProbeHeads, build_model, eval_probe, train_probe
from faof.tokenizer import CharTokenizer
from faof.train import autocast_context, loss_fn, lr_at_step, pick_device, pick_dtype, set_seed

# (anchor_weight, order_free_weight, bridge_weight) turned on cumulatively.
VARIANTS = {
    "A_baseline": (0.0, 0.0, 0.0),
    "B_anchor": (0.2, 0.0, 0.0),
    "C_anchor_order": (0.2, 0.1, 0.0),
    "D_full": (0.2, 0.1, 0.2),
}
METRIC_KEYS = ["next_loss", "anchor_top5@2", "anchor_top5@4", "anchor_top5@8", "anchor_top5@16", "order_recall"]
PROBE_SEED = 1234  # fixed: identical probe procedure for every backbone


def flat_metrics(m: dict) -> dict:
    out = {"next_loss": m["next_loss"], "order_recall": m["order_recall"]}
    for k, v in m["anchor_top5"].items():
        out[f"anchor_top5@{k}"] = v
    return out


def train_backbone(cfg, seed, device, dtype, vocab, steps):
    set_seed(seed)
    model = build_model(cfg, vocab, device, dtype)
    model.train()
    ds = BinaryTokenDataset(cfg.train_bin, cfg.block_size, cfg.extra_tokens, vocab, device)
    opt = AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    for step in range(1, steps + 1):
        full_ids = ds.sample_batch(cfg.batch_size)
        with autocast_context(device, dtype):
            loss, _ = loss_fn(model, cfg, full_ids)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        lr = lr_at_step(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    return model


def probe_backbone(model, cfg, device, dtype, vocab, probe_steps, eval_batches):
    torch.manual_seed(PROBE_SEED)
    np.random.seed(PROBE_SEED)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    probe = ProbeHeads(cfg, vocab).to(device=device, dtype=dtype)
    train_ds = BinaryTokenDataset(cfg.train_bin, cfg.block_size, cfg.extra_tokens, vocab, device)
    val_ds = BinaryTokenDataset(cfg.val_bin, cfg.block_size, cfg.extra_tokens, vocab, device)
    train_probe(model, probe, cfg, train_ds, device, dtype, probe_steps, 1e-3)
    return eval_probe(model, probe, cfg, val_ds, device, dtype, eval_batches)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="configs/wiki_ablation/A_baseline.json", help="base config for arch + data paths")
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--probe-steps", type=int, default=400)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1337, 2024, 7])
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS),
                        help="subset of variants to run (default: all four)")
    parser.add_argument("--out", default="runs/probe_ablation")
    args = parser.parse_args()
    variants = {k: VARIANTS[k] for k in args.variants}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_path = out / "rows.jsonl"
    rows_path.write_text("", encoding="utf-8")

    base = TrainConfig.from_json(args.base)
    device = pick_device(base.device)
    dtype = pick_dtype(base.dtype)
    tok = CharTokenizer.load(base.tokenizer_path)
    vocab = tok.vocab_size
    n_runs = len(variants) * len(args.seeds)
    print(f"base={args.base} device={device} dtype={dtype} vocab={vocab} "
          f"steps={args.steps} seeds={args.seeds} runs={n_runs}", flush=True)

    rows = []
    t_start = time.time()
    for variant, (aw, ow, bw) in variants.items():
        for seed in args.seeds:
            cfg = TrainConfig.from_json(args.base)
            cfg.anchor_weight, cfg.order_free_weight, cfg.bridge_weight = aw, ow, bw
            cfg.max_steps, cfg.warmup_steps = args.steps, args.warmup
            t0 = time.time()
            try:
                model = train_backbone(cfg, seed, device, dtype, vocab, args.steps)
                metrics = flat_metrics(
                    probe_backbone(model, cfg, device, dtype, vocab, args.probe_steps, args.eval_batches)
                )
                del model
                if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
                    torch.mps.empty_cache()
                row = {"variant": variant, "seed": seed, "sec": round(time.time() - t0, 1), **metrics}
            except Exception as e:  # keep the sweep alive; record the failure
                row = {"variant": variant, "seed": seed, "error": repr(e)}
            rows.append(row)
            with rows_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[{len(rows)}/{n_runs}] {variant} seed={seed} {row.get('sec','ERR')}s "
                  f"next={row.get('next_loss', float('nan')):.4f} "
                  f"a@8={row.get('anchor_top5@8', float('nan')):.4f} "
                  f"(elapsed {(time.time()-t_start)/60:.1f}m)", flush=True)

    agg = {}
    for variant in variants:
        vals = {k: [r[k] for r in rows if r["variant"] == variant and k in r] for k in METRIC_KEYS}
        agg[variant] = {
            k: {"mean": float(np.mean(v)), "std": float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
            for k, v in vals.items()
        }
    (out / "aggregate.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n================ AGGREGATE (mean +/- std over seeds) ================", flush=True)
    print("variant".ljust(16) + "".join(k.rjust(20) for k in METRIC_KEYS), flush=True)
    for variant in variants:
        line = variant.ljust(16)
        for k in METRIC_KEYS:
            s = agg[variant][k]
            line += f"{s['mean']:.4f}±{s['std']:.4f}".rjust(20)
        print(line, flush=True)
    if "A_baseline" in variants and "D_full" in variants:
        a, d = agg["A_baseline"], agg["D_full"]
        print("\n---- D_full vs A_baseline (Δ = D − A) ----", flush=True)
        for k in METRIC_KEYS:
            delta = d[k]["mean"] - a[k]["mean"]
            noise = (a[k]["std"] ** 2 + d[k]["std"] ** 2) ** 0.5
            sig = "SIGNIFICANT" if noise > 0 and abs(delta) > 2 * noise else "within noise"
            print(f"  {k:16s} Δ={delta:+.4f}  (±{noise:.4f} combined)  {sig}", flush=True)
    print(f"\ntotal wall time: {(time.time() - t_start)/60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
