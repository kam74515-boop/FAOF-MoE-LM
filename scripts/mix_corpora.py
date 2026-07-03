from __future__ import annotations

import argparse
import random
from pathlib import Path


def load_lines(path: Path, max_docs: int) -> list[str]:
    lines = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(line)
            if max_docs and len(lines) >= max_docs:
                break
    return lines


def parse_source(spec: str) -> tuple[Path, float]:
    if ":" not in spec:
        raise ValueError("source must look like path:weight")
    path, weight = spec.rsplit(":", 1)
    return Path(path), float(weight)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Make a small weighted line-level corpus from cleaned text files."
    )
    parser.add_argument("--source", action="append", required=True, help="path:weight")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-docs", type=int, default=100_000)
    parser.add_argument("--per-source-read", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(args.seed)
    sources = [(path, weight, load_lines(path, args.per_source_read)) for path, weight in map(parse_source, args.source)]
    total_weight = sum(weight for _, weight, lines in sources if lines)
    if total_weight <= 0:
        raise ValueError("no non-empty sources")

    mixed = []
    for path, weight, lines in sources:
        if not lines:
            continue
        take = max(1, int(args.max_docs * weight / total_weight))
        if take <= len(lines):
            mixed.extend(random.sample(lines, take))
        else:
            mixed.extend(random.choice(lines) for _ in range(take))
        print(f"{path}: loaded={len(lines):,} target={take:,}")
    random.shuffle(mixed)
    mixed = mixed[: args.max_docs]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for line in mixed:
            f.write(line + "\n")
    print(f"wrote {len(mixed):,} docs to {out}")


if __name__ == "__main__":
    main()

