#!/usr/bin/env bash
# Overnight "give the auxiliary losses their best shot at scale" run.
#
# Re-tokenizes the FULL zhwiki_small shard (no --max-lines cap), then trains and
# probe-compares A_baseline vs D_full at a large step count on de-starved data.
# This is the expensive confirmation of the moderate sweep: if the FAOF bundle
# shows no probe benefit here either, the design needs rethinking, not more compute.
#
# Uses its own data/config dirs (bin_epoch / wiki_epoch) so it never clobbers the
# bins a concurrently-running sweep has memory-mapped.
#
# Usage:  bash scripts/run_full_epoch.sh [STEPS] [SEED]
#   STEPS defaults to 30000 (~3.4h/run on an M5). ~88000 ≈ one full epoch over the
#   full shard (~180M tokens); at that point expect ~7h for the A+D pair.
set -euo pipefail

STEPS="${1:-30000}"
SEED="${2:-1337}"
DATA=data/zhwiki_small
BIN=$DATA/bin_epoch
TOK=$DATA/tokenizer_epoch.json
CFG=configs/wiki_epoch.json

echo ">> tokenizing the full shard (all lines) -> $BIN"
uv run python scripts/build_char_tokenizer.py --input "$DATA/zhwiki_small.txt" --out "$TOK" --vocab-size 8000
uv run python scripts/tokenize_text.py --input "$DATA/zhwiki_small.txt" --tokenizer "$TOK" --out-dir "$BIN" --val-ratio 0.02

echo ">> writing base config -> $CFG"
uv run python - "$BIN" "$TOK" "$CFG" <<'PY'
import json, sys
bin_dir, tok, cfg_path = sys.argv[1:4]
c = json.load(open('configs/wiki_ablation/A_baseline.json'))
c.update(run_name='wiki_epoch', out_dir='runs/wiki_epoch',
         train_bin=f'{bin_dir}/train.bin', val_bin=f'{bin_dir}/val.bin', tokenizer_path=tok)
json.dump(c, open(cfg_path, 'w'), indent=2, ensure_ascii=False)
print('wrote', cfg_path)
PY

echo ">> probe-comparing A_baseline vs D_full  (steps=$STEPS seed=$SEED)"
uv run python scripts/run_probe_ablation.py \
  --base "$CFG" --steps "$STEPS" --warmup 1000 --seeds "$SEED" \
  --variants A_baseline D_full --out runs/probe_ablation_epoch
