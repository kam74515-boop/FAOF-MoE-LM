#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-data/zhwiki}"
mkdir -p "$OUT_DIR"

URL="https://dumps.wikimedia.org/zhwiki/latest/zhwiki-latest-pages-articles.xml.bz2"
DEST="$OUT_DIR/zhwiki-latest-pages-articles.xml.bz2"

if [[ ! -f "$DEST" ]]; then
  curl -L "$URL" -o "$DEST"
else
  echo "$DEST already exists"
fi

cat <<'MSG'

Downloaded the Chinese Wikipedia articles dump.

Next extraction option:

  uv add wikiextractor
  uv run python -m wikiextractor.WikiExtractor \
    --json --no-templates \
    -o data/zhwiki/extracted \
    data/zhwiki/zhwiki-latest-pages-articles.xml.bz2

Then convert extracted JSON lines to plain text before tokenization, or use your
preferred MediaWiki dump cleaner.
MSG

