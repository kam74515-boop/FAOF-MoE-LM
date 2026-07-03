from __future__ import annotations

import argparse
from pathlib import Path

from faof.data import split_and_write
from faof.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-lines", type=int, default=0)
    args = parser.parse_args()
    tok = CharTokenizer.load(args.tokenizer)
    ids: list[int] = []
    for name in args.input:
        with Path(name).open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if args.max_lines and i >= args.max_lines:
                    break
                line = line.strip()
                if line:
                    ids.extend(tok.encode(line))
    train_path, val_path = split_and_write(ids, args.out_dir, args.val_ratio, tok.vocab_size)
    print(f"tokens={len(ids):,} train={train_path} val={val_path} vocab={tok.vocab_size}")


if __name__ == "__main__":
    main()
