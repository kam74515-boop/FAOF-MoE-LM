from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


class CharTokenizer:
    pad_token = "<pad>"
    unk_token = "<unk>"
    bos_token = "<bos>"
    eos_token = "<eos>"

    def __init__(self, stoi: dict[str, int]):
        self.stoi = stoi
        self.itos = {v: k for k, v in stoi.items()}
        self.pad_id = stoi[self.pad_token]
        self.unk_id = stoi[self.unk_token]
        self.bos_id = stoi[self.bos_token]
        self.eos_id = stoi[self.eos_token]

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    @classmethod
    def train(cls, texts: list[str], vocab_size: int) -> "CharTokenizer":
        specials = [cls.pad_token, cls.unk_token, cls.bos_token, cls.eos_token]
        counter: Counter[str] = Counter()
        for text in texts:
            counter.update(ch for ch in text if not ch.isspace())
        keep = max(0, vocab_size - len(specials))
        chars = [ch for ch, _ in counter.most_common(keep)]
        stoi = {tok: i for i, tok in enumerate(specials + chars)}
        return cls(stoi)

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        ids = [self.stoi.get(ch, self.unk_id) for ch in text if not ch.isspace()]
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def decode(self, ids: list[int]) -> str:
        out = []
        for idx in ids:
            token = self.itos.get(int(idx), self.unk_token)
            if token in {self.pad_token, self.bos_token, self.eos_token}:
                continue
            out.append(token)
        return "".join(out)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps({"stoi": self.stoi}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "CharTokenizer":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: int(v) for k, v in raw["stoi"].items()})

