#!/usr/bin/env bash
set -euo pipefail

CONFIG_DIR="${1:-configs/ablation_smoke}"

uv run python -m faof.train --config "$CONFIG_DIR/A_baseline.json"
uv run python -m faof.train --config "$CONFIG_DIR/B_anchor.json"
uv run python -m faof.train --config "$CONFIG_DIR/C_anchor_order.json"
uv run python -m faof.train --config "$CONFIG_DIR/D_anchor_order_bridge.json"

for name in A_baseline B_anchor C_anchor_order D_anchor_order_bridge; do
  echo "==== $name ===="
  ckpt="$(uv run python -c 'import json, sys; cfg=json.load(open(sys.argv[1])); print(cfg["out_dir"] + "/ckpt_last.pt")' "$CONFIG_DIR/$name.json")"
  uv run python -m faof.eval --config "$CONFIG_DIR/$name.json" --ckpt "$ckpt" --batches 5
done
