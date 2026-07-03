from __future__ import annotations

import argparse
from pathlib import Path

from faof.tokenizer import CharTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--vocab-size", type=int, default=8000)
    parser.add_argument("--max-lines", type=int, default=2_000_000)
    args = parser.parse_args()
    texts = []
    for name in args.input:
        with Path(name).open(encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= args.max_lines:
                    break
                line = line.strip()
                if line:
                    texts.append(line)
    tok = CharTokenizer.train(texts, args.vocab_size)
    tok.save(args.out)
    print(f"wrote {args.out} vocab_size={tok.vocab_size}")


if __name__ == "__main__":
    main()

