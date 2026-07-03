from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional


@dataclass
class TrainConfig:
    run_name: str
    train_bin: str
    val_bin: str
    tokenizer_path: str
    out_dir: str
    init_from: Optional[str] = None
    resume_from: Optional[str] = None
    device: str = "auto"
    dtype: str = "float32"
    activation_checkpointing: bool = False
    save_optimizer: bool = True
    seed: int = 1337

    block_size: int = 1024
    batch_size: int = 16
    grad_accum_steps: int = 1
    max_steps: int = 1000
    eval_interval: int = 100
    log_interval: int = 10
    save_interval: int = 1000

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100

    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0

    offsets: tuple[int, ...] = (2, 4, 8, 16)
    future_window: int = 32
    order_positions: int = 16
    bridge_positions: int = 8
    bridge_offsets: tuple[int, ...] = (4, 8, 16)
    soft_anchor_top_m: int = 128

    next_weight: float = 1.0
    anchor_weight: float = 0.2
    order_free_weight: float = 0.1
    bridge_weight: float = 0.2
    order_free_neg_weight: float = 0.02

    @property
    def extra_tokens(self) -> int:
        return max(max(self.offsets), self.future_window)

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        for key in ("offsets", "bridge_offsets"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return cls(**raw)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["offsets"] = list(self.offsets)
        data["bridge_offsets"] = list(self.bridge_offsets)
        return data
