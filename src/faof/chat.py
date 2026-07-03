from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .model import FAOFModel, ModelConfig
from .tokenizer import CharTokenizer
from .train import pick_device, pick_dtype


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 0 or k >= logits.shape[-1]:
        return logits
    values, _ = torch.topk(logits, k)
    cutoff = values[..., -1, None]
    return logits.masked_fill(logits < cutoff, float("-inf"))


def sample_from_logits(logits: torch.Tensor, temperature: float, top_k: int) -> torch.Tensor:
    logits = logits.float()
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = top_k_filter(logits / temperature, top_k)
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def sample(
    model: FAOFModel,
    tok: CharTokenizer,
    cfg: TrainConfig,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    device: torch.device,
) -> str:
    ids = tok.encode(prompt, add_eos=False)
    x = torch.tensor(ids, device=device, dtype=torch.long)[None, :]
    for _ in range(max_new_tokens):
        x_cond = x[:, -cfg.block_size :]
        logits = model(x_cond)["next_logits"][:, -1, :].float()
        next_id = sample_from_logits(logits, temperature, top_k)
        x = torch.cat([x, next_id], dim=1)
    return tok.decode(x[0].tolist())


@torch.no_grad()
def sample_faof(
    model: FAOFModel,
    tok: CharTokenizer,
    cfg: TrainConfig,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    anchor_offset: int,
    device: torch.device,
) -> tuple[str, list[str]]:
    if str(anchor_offset) not in model.anchor_heads:
        raise ValueError(f"anchor_offset must be one of {sorted(model.anchor_heads.keys())}")
    if anchor_offset not in cfg.bridge_offsets:
        raise ValueError(f"anchor_offset must be one of bridge_offsets={cfg.bridge_offsets}")

    ids = tok.encode(prompt, add_eos=False)
    x = torch.tensor(ids, device=device, dtype=torch.long)[None, :]
    anchors = []
    while x.shape[1] - len(ids) < max_new_tokens:
        x_cond = x[:, -cfg.block_size :]
        pos = torch.tensor([x_cond.shape[1] - 1], device=device, dtype=torch.long)
        off = torch.tensor([anchor_offset], device=device, dtype=torch.long)
        out = model(x_cond, bridge_positions=pos, bridge_offsets=off)

        bridge_logits = out["bridge_logits"][:, 0, : anchor_offset - 1, :]
        new_ids = []
        remaining = max_new_tokens - (x.shape[1] - len(ids))
        for i in range(min(anchor_offset - 1, remaining)):
            next_id = sample_from_logits(bridge_logits[:, i, :], temperature, top_k)
            new_ids.append(next_id)

        remaining = max_new_tokens - (x.shape[1] - len(ids)) - len(new_ids)
        if remaining > 0:
            anchor_logits = out["anchor_logits"][str(anchor_offset)][:, pos.item(), :]
            anchor_id = sample_from_logits(anchor_logits, temperature, top_k)
            anchors.append(tok.decode(anchor_id[0].tolist()))
            new_ids.append(anchor_id)

        if not new_ids:
            break
        x = torch.cat([x, *new_ids], dim=1)

    return tok.decode(x[0].tolist()), anchors


def load_model(config_path: str, ckpt_path: str):
    cfg = TrainConfig.from_json(config_path)
    device = pick_device(cfg.device)
    dtype = pick_dtype(cfg.dtype)
    tok = CharTokenizer.load(cfg.tokenizer_path)
    model_cfg = ModelConfig(
        vocab_size=tok.vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        dropout=0.0,
        offsets=cfg.offsets,
        bridge_offsets=cfg.bridge_offsets,
        soft_anchor_top_m=cfg.soft_anchor_top_m,
        activation_checkpointing=False,
    )
    model = FAOFModel(model_cfg).to(device=device, dtype=dtype)
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return cfg, tok, model, device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/local_complete_18m_epoch.json")
    parser.add_argument("--ckpt", default="runs/local_complete_18m_epoch/ckpt_last.pt")
    parser.add_argument("--max-new-tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=60)
    parser.add_argument("--mode", choices=("next", "faof"), default="next")
    parser.add_argument("--anchor-offset", type=int, default=8)
    parser.add_argument("--show-anchors", action="store_true")
    parser.add_argument("--system-prefix", default="")
    args = parser.parse_args()

    cfg, tok, model, device = load_model(args.config, args.ckpt)
    print("FAOF local model loaded. 输入 /exit 退出，/help 查看命令。", flush=True)
    print("提示：这是 Wiki 小模型，适合续写，不是指令微调聊天模型。", flush=True)

    while True:
        try:
            line = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            break
        if not line:
            continue
        if line in {"/exit", "/quit", "exit", "quit"}:
            print("bye")
            break
        if line == "/help":
            print("输入任意中文作为 prompt，模型会继续生成。退出请输入 /exit。")
            print(f"当前 max_new_tokens={args.max_new_tokens} temperature={args.temperature} top_k={args.top_k}")
            continue
        prompt = args.system_prefix + line
        if args.mode == "faof":
            text, anchors = sample_faof(
                model,
                tok,
                cfg,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                args.anchor_offset,
                device,
            )
            if args.show_anchors:
                print(f"锚点> {' / '.join(a or '<unk>' for a in anchors)}", flush=True)
        else:
            text = sample(
                model,
                tok,
                cfg,
                prompt,
                args.max_new_tokens,
                args.temperature,
                args.top_k,
                device,
            )
        reply = text[len(prompt) :]
        print(f"模型> {reply}", flush=True)


if __name__ == "__main__":
    main()
