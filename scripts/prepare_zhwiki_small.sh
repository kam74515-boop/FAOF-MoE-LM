#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/zhwiki_small}"
mkdir -p "$OUT_DIR"

URL="https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles1.xml-p1p187712.bz2"
DEST="$OUT_DIR/zhwiki-latest-pages-articles1.xml-p1p187712.bz2"
TEXT="$OUT_DIR/zhwiki_small.txt"

if [[ ! -f "$DEST" ]]; then
  curl -L "$URL" -o "$DEST"
else
  echo "$DEST already exists"
fi

uv run python scripts/extract_wiki_xml.py \
  --input "$DEST" \
  --out "$TEXT" \
  --max-docs 50000 \
  --min-chars 120

echo "wrote $TEXT"

