from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


class BinaryTokenDataset:
    def __init__(
        self,
        path: str | Path,
        block_size: int,
        extra_tokens: int,
        vocab_size: int,
        device: torch.device,
    ):
        self.path = Path(path)
        dtype = np.uint16 if vocab_size <= 65535 else np.uint32
        self.tokens = np.memmap(self.path, dtype=dtype, mode="r")
        self.block_size = block_size
        self.extra_tokens = extra_tokens
        self.sample_len = block_size + extra_tokens
        self.device = device
        if len(self.tokens) <= self.sample_len + 1:
            raise ValueError(f"{self.path} has too few tokens: {len(self.tokens)}")

    def sample_batch(self, batch_size: int) -> torch.Tensor:
        max_start = len(self.tokens) - self.sample_len - 1
        starts = np.random.randint(0, max_start, size=batch_size)
        batch = np.stack([self.tokens[s : s + self.sample_len] for s in starts])
        return torch.from_numpy(batch.astype(np.int64)).to(self.device)


def split_and_write(
    ids: list[int],
    out_dir: str | Path,
    val_ratio: float,
    vocab_size: int,
) -> tuple[Path, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_val = max(1, int(len(ids) * val_ratio))
    n_train = max(1, len(ids) - n_val)
    dtype = np.uint16 if vocab_size <= 65535 else np.uint32
    train = np.asarray(ids[:n_train], dtype=dtype)
    val = np.asarray(ids[n_train:], dtype=dtype)
    train_path = out / "train.bin"
    val_path = out / "val.bin"
    train.tofile(train_path)
    val.tofile(val_path)
    return train_path, val_path

