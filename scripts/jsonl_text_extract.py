from __future__ import annotations

import argparse
import json
from pathlib import Path


def iter_files(paths: list[str]):
    for name in paths:
        path = Path(name)
        if path.is_dir():
            yield from sorted(p for p in path.rglob("*") if p.is_file())
        else:
            yield path


def clean(text: str) -> str:
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(("==", "{{", "|", "[[Category:")):
            continue
        lines.append(line)
    return "".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract plain text from WikiExtractor-style JSONL files."
    )
    parser.add_argument("--input", required=True, nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--field", default="text")
    parser.add_argument("--min-chars", type=int, default=80)
    parser.add_argument("--max-docs", type=int, default=0)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w", encoding="utf-8") as fout:
        for path in iter_files(args.input):
            with path.open(encoding="utf-8", errors="ignore") as fin:
                for line in fin:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                        text = clean(str(obj.get(args.field, "")))
                    except json.JSONDecodeError:
                        text = clean(line)
                    if len(text) < args.min_chars:
                        continue
                    fout.write(text + "\n")
                    written += 1
                    if args.max_docs and written >= args.max_docs:
                        print(f"wrote {written:,} docs to {out}")
                        return
    print(f"wrote {written:,} docs to {out}")


if __name__ == "__main__":
    main()

