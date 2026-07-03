from __future__ import annotations

import argparse
import bz2
import re
import xml.etree.ElementTree as ET
from pathlib import Path


TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
REF_RE = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S)
TAG_RE = re.compile(r"<[^>]+>")
FILE_LINK_RE = re.compile(r"\[\[(?:File|Image|文件|檔案|Category|分类|分類):[^\]]+\]\]", re.I)
EXTERNAL_LINK_RE = re.compile(r"\[(https?://\S+)(?:\s+([^\]]+))?\]")
INTERNAL_LINK_RE = re.compile(r"\[\[([^\]|]+)\|([^\]]+)\]\]|\[\[([^\]]+)\]\]")


def strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def clean_wiki_text(text: str) -> str:
    text = REF_RE.sub("", text)
    for _ in range(4):
        new = TEMPLATE_RE.sub("", text)
        if new == text:
            break
        text = new
    text = FILE_LINK_RE.sub("", text)
    text = EXTERNAL_LINK_RE.sub(lambda m: m.group(2) or "", text)
    text = INTERNAL_LINK_RE.sub(lambda m: m.group(2) or m.group(3) or m.group(1), text)
    text = TAG_RE.sub("", text)
    text = re.sub(r"'{2,5}", "", text)
    text = re.sub(r"^=+\s*(.*?)\s*=+$", r"\1。", text, flags=re.M)
    text = re.sub(r"^[*#;:]+", "", text, flags=re.M)
    text = re.sub(r"\{\||\|\}|\|[-+]?|!", "", text)
    text = re.sub(r"[ \t]+", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) < 12:
            continue
        if line.startswith(("REDIRECT", "#重定向", "#REDIRECT")):
            continue
        lines.append(line)
    return "".join(lines)


def find_child_text(elem: ET.Element, name: str) -> str:
    for child in elem.iter():
        if strip_namespace(child.tag) == name:
            return child.text or ""
    return ""


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract rough plain text from a zhwiki XML bz2 dump.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-docs", type=int, default=0)
    parser.add_argument("--min-chars", type=int, default=120)
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    opener = bz2.open if args.input.endswith(".bz2") else open
    with opener(args.input, "rb") as fin, out.open("w", encoding="utf-8") as fout:
        context = ET.iterparse(fin, events=("end",))
        for _, elem in context:
            if strip_namespace(elem.tag) != "page":
                continue
            ns = find_child_text(elem, "ns")
            if ns != "0":
                elem.clear()
                continue
            title = find_child_text(elem, "title")
            raw = find_child_text(elem, "text")
            text = clean_wiki_text(raw)
            if len(text) >= args.min_chars:
                fout.write(f"{title}。{text}\n")
                written += 1
                if args.max_docs and written >= args.max_docs:
                    break
            elem.clear()
    print(f"wrote {written:,} docs to {out}")


if __name__ == "__main__":
    main()

