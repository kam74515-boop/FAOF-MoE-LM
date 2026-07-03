from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class ModelConfig:
    vocab_size: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    offsets: tuple[int, ...]
    bridge_offsets: tuple[int, ...]
    soft_anchor_top_m: int
    activation_checkpointing: bool = False


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=False)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.proj(y)


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        hidden = int(8 * cfg.n_embd / 3)
        hidden = ((hidden + 255) // 256) * 256
        self.w1 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embd, bias=False)
        self.w3 = nn.Linear(cfg.n_embd, hidden, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class FAOFModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_embedding = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)

        self.next_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.next_head.weight = self.token_embedding.weight
        self.anchor_heads = nn.ModuleDict(
            {str(k): nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False) for k in cfg.offsets}
        )
        self.order_free_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.span_embedding = nn.Embedding(max(cfg.bridge_offsets) + 1, cfg.n_embd)
        self.bridge_pos_embedding = nn.Embedding(max(cfg.bridge_offsets), cfg.n_embd)
        self.bridge_mlp = nn.Sequential(
            nn.Linear(cfg.n_embd * 4, cfg.n_embd),
            nn.GELU(),
            nn.Linear(cfg.n_embd, cfg.n_embd),
        )
        self.bridge_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        order_positions: torch.Tensor | None = None,
        bridge_positions: torch.Tensor | None = None,
        bridge_offsets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        bsz, seqlen = input_ids.shape
        if seqlen > self.cfg.block_size:
            raise ValueError(f"input length {seqlen} > block_size {self.cfg.block_size}")
        pos = torch.arange(seqlen, device=input_ids.device)
        x = self.token_embedding(input_ids) + self.pos_embedding(pos)[None, :, :]
        x = self.drop(x)
        for block in self.blocks:
            if self.cfg.activation_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        h = self.ln_f(x)
        anchor_logits = {k: head(h) for k, head in self.anchor_heads.items()}
        out: dict[str, torch.Tensor | dict[str, torch.Tensor]] = {
            "hidden": h,
            "next_logits": self.next_head(h),
            "anchor_logits": anchor_logits,
        }
        if order_positions is not None:
            out["order_logits"] = self.order_free_head(h[:, order_positions])
        if bridge_positions is not None and bridge_offsets is not None:
            out["bridge_logits"] = self.bridge_forward(
                h, input_ids, anchor_logits, bridge_positions, bridge_offsets
            )
        return out

    def soft_anchor(self, logits: torch.Tensor) -> torch.Tensor:
        top_m = min(self.cfg.soft_anchor_top_m, logits.shape[-1])
        vals, idx = logits.float().topk(top_m, dim=-1)
        probs = torch.softmax(vals, dim=-1).to(self.token_embedding.weight.dtype)
        emb = self.token_embedding(idx)
        return (probs.unsqueeze(-1) * emb).sum(dim=-2)

    def bridge_forward(
        self,
        h: torch.Tensor,
        input_ids: torch.Tensor,
        anchor_logits: dict[str, torch.Tensor],
        bridge_positions: torch.Tensor,
        bridge_offsets: torch.Tensor,
    ) -> torch.Tensor:
        bsz = h.shape[0]
        pieces = []
        for pos, off in zip(bridge_positions.tolist(), bridge_offsets.tolist(), strict=True):
            left_h = h[:, pos]
            left_emb = self.token_embedding(input_ids[:, pos])
            right_soft = self.soft_anchor(anchor_logits[str(int(off))][:, pos])
            span_id = torch.full((bsz,), int(off), device=h.device, dtype=torch.long)
            span_emb = self.span_embedding(span_id)
            cond = self.bridge_mlp(torch.cat([left_h, left_emb, right_soft, span_emb], dim=-1))
            max_len = max(self.cfg.bridge_offsets) - 1
            rel = torch.arange(max_len, device=h.device)
            bridge_h = cond[:, None, :] + self.bridge_pos_embedding(rel)[None, :, :]
            pieces.append(self.bridge_head(bridge_h))
        return torch.stack(pieces, dim=1)
