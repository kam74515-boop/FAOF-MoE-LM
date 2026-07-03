from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path


RUNS = {
    "A_baseline": {
        "anchor_weight": 0.0,
        "order_free_weight": 0.0,
        "bridge_weight": 0.0,
        "init_from": None,
    },
    "B_anchor": {
        "anchor_weight": 0.2,
        "order_free_weight": 0.0,
        "bridge_weight": 0.0,
    },
    "C_anchor_order": {
        "anchor_weight": 0.2,
        "order_free_weight": 0.1,
        "bridge_weight": 0.0,
    },
    "D_anchor_order_bridge": {
        "anchor_weight": 0.2,
        "order_free_weight": 0.1,
        "bridge_weight": 0.2,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--baseline-ckpt", default=None)
    parser.add_argument("--max-steps", type=int, default=0)
    args = parser.parse_args()

    base = json.loads(Path(args.base).read_text(encoding="utf-8"))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    baseline_ckpt = args.baseline_ckpt or str(Path(args.run_root) / "A_baseline" / "ckpt_last.pt")

    for name, overrides in RUNS.items():
        cfg = deepcopy(base)
        cfg["run_name"] = name
        cfg["out_dir"] = str(Path(args.run_root) / name)
        cfg["resume_from"] = None
        cfg["init_from"] = None if name == "A_baseline" else baseline_ckpt
        cfg.update({k: v for k, v in overrides.items() if v is not None})
        if args.max_steps:
            cfg["max_steps"] = args.max_steps
            cfg["save_interval"] = args.max_steps
            cfg["eval_interval"] = max(1, args.max_steps // 2)
        path = out_dir / f"{name}.json"
        path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()

