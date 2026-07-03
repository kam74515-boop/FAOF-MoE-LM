# FAOF-LM: Future-Anchored Order-Free Language Model

**English** | [中文](#中文文档)

Minimal training code for a small Chinese **Future-Anchored Order-Free** (FAOF) language model. The first implementation is intentionally **dense, not MoE**, so the core idea can be validated before adding routing complexity.

## What This Trains

The model is a causal decoder with auxiliary heads:

- **`next`**: standard next-token prediction.
- **`anchor`**: predicts future tokens at offsets `2, 4, 8, 16`.
- **`order_free`**: predicts the unordered *set* of tokens in a future window.
- **`bridge`**: uses the current hidden state plus soft future-anchor embeddings to reconstruct the span between the current position and a future anchor.

The default smoke config is tiny and only proves that the data, losses, and evaluation loop work. Use `configs/mvp_120m.json` as the starting point for a real 100M-scale run on a GPU machine.

## Quick Smoke Test

```bash
uv sync
uv run python scripts/make_toy_corpus.py --out data/toy.txt --docs 4000
uv run python scripts/build_char_tokenizer.py --input data/toy.txt --out data/tokenizer.json --vocab-size 3000
uv run python scripts/tokenize_text.py --input data/toy.txt --tokenizer data/tokenizer.json --out-dir data/bin --val-ratio 0.02
uv run python -m faof.train --config configs/smoke.json
uv run python -m faof.eval --config configs/smoke.json --ckpt runs/smoke/ckpt_last.pt
```

## Interactive Generation

```bash
# ordinary autoregressive sampling
uv run python -m faof.chat --config <cfg.json> --ckpt <ckpt.pt> --mode next

# anchor-first FAOF sampling: bridge-decode toward a sampled future anchor
uv run python -m faof.chat --config <cfg.json> --ckpt <ckpt.pt> --mode faof --show-anchors
```

## Chinese Wiki Data

For the smallest real corpus, use Chinese Wikipedia plus a small amount of CLUECorpusSmall or another open Chinese corpus.

Recommended small recipe:

- 70% Chinese Wikipedia extracted plain text.
- 20% CLUECorpusSmall.
- 10% high-quality web sample such as Ultra-FineWeb zh.

Download the Chinese Wikipedia articles dump:

```bash
bash scripts/prepare_zhwiki.sh data/zhwiki
```

The script downloads `zhwiki-latest-pages-articles.xml.bz2`. Extract it with WikiExtractor or your preferred MediaWiki dump cleaner, then pass the resulting plain text to `scripts/build_char_tokenizer.py` and `scripts/tokenize_text.py`.

For a much smaller first real-data run, download only the first articles shard:

```bash
bash scripts/prepare_zhwiki_small.sh data/zhwiki_small
uv run python scripts/build_char_tokenizer.py --input data/zhwiki_small/zhwiki_small.txt --out data/zhwiki_small/tokenizer.json --vocab-size 8000 --max-lines 2000
uv run python scripts/tokenize_text.py --input data/zhwiki_small/zhwiki_small.txt --tokenizer data/zhwiki_small/tokenizer.json --out-dir data/zhwiki_small/bin --val-ratio 0.02 --max-lines 2000
uv run python -m faof.train --config configs/wiki_smoke.json
```

## MVP Ablation

Train the same base checkpoint with these loss weights:

| Run | `anchor_weight` | `order_free_weight` | `bridge_weight` |
| --- | ---: | ---: | ---: |
| A baseline | 0.0 | 0.0 | 0.0 |
| B anchor | 0.2 | 0.0 | 0.0 |
| C anchor+order | 0.2 | 0.1 | 0.0 |
| D anchor+order+bridge | 0.2 | 0.1 | 0.2 |

Success criteria:

- Anchor `top5@8` and `top5@16` improve over a baseline probe.
- Order-free future token recall improves over anchor-only.
- Bridge token accuracy improves over next-token continuation for infill spans.
- Validation next-token loss does not degrade by more than about 3%.

Generate and run a toy ablation:

```bash
uv run python scripts/make_ablation_configs.py \
  --base configs/smoke.json \
  --out-dir configs/ablation_smoke \
  --run-root runs/ablation_smoke \
  --max-steps 60

bash scripts/run_ablation.sh configs/ablation_smoke
```

## Cleaning WikiExtractor Output

If you extract the Wikipedia dump with WikiExtractor JSON output:

```bash
uv run python scripts/jsonl_text_extract.py \
  --input data/zhwiki/extracted \
  --out data/zhwiki/zhwiki.txt \
  --min-chars 80
```

You can then build a tiny mixed corpus:

```bash
uv run python scripts/mix_corpora.py \
  --source data/zhwiki/zhwiki.txt:0.7 \
  --source data/clue/clue.txt:0.2 \
  --source data/web/web_sample.txt:0.1 \
  --out data/mixed_zh_small.txt \
  --max-docs 100000
```

## License

[MIT](LICENSE)

---

# 中文文档

[English](#faof-lm-future-anchored-order-free-language-model) | **中文**

一个小型中文 **未来锚定·顺序无关**（Future-Anchored Order-Free，FAOF）语言模型的极简训练代码。第一版实现刻意采用 **dense 架构而非 MoE**，以便在引入路由复杂度之前先验证核心思想。

## 训练目标

模型是一个带辅助预测头的因果解码器：

- **`next`**：标准的下一 token 预测。
- **`anchor`**：预测偏移 `2, 4, 8, 16` 处的未来 token。
- **`order_free`**：预测未来窗口内 token 的无序*集合*。
- **`bridge`**：利用当前隐状态加上软未来锚点嵌入，重建当前位置与未来锚点之间的跨度。

默认 smoke 配置非常小，仅用于验证数据、损失和评估流程能跑通。在 GPU 机器上做真正的 1 亿参数级训练时，请以 `configs/mvp_120m.json` 为起点。

## 快速冒烟测试

```bash
uv sync
uv run python scripts/make_toy_corpus.py --out data/toy.txt --docs 4000
uv run python scripts/build_char_tokenizer.py --input data/toy.txt --out data/tokenizer.json --vocab-size 3000
uv run python scripts/tokenize_text.py --input data/toy.txt --tokenizer data/tokenizer.json --out-dir data/bin --val-ratio 0.02
uv run python -m faof.train --config configs/smoke.json
uv run python -m faof.eval --config configs/smoke.json --ckpt runs/smoke/ckpt_last.pt
```

## 交互式生成

```bash
# 普通自回归采样
uv run python -m faof.chat --config <cfg.json> --ckpt <ckpt.pt> --mode next

# 锚点优先的 FAOF 采样：向采样出的未来锚点做 bridge 解码
uv run python -m faof.chat --config <cfg.json> --ckpt <ckpt.pt> --mode faof --show-anchors
```

## 中文维基数据

最小可用真实语料建议使用中文维基百科，再混入少量 CLUECorpusSmall 或其他开放中文语料。

推荐的小配方：

- 70% 中文维基百科抽取的纯文本。
- 20% CLUECorpusSmall。
- 10% 高质量网络样本，例如 Ultra-FineWeb 中文。

下载中文维基百科条目 dump：

```bash
bash scripts/prepare_zhwiki.sh data/zhwiki
```

该脚本下载 `zhwiki-latest-pages-articles.xml.bz2`。用 WikiExtractor 或你惯用的 MediaWiki dump 清洗工具抽取后，将得到的纯文本传给 `scripts/build_char_tokenizer.py` 和 `scripts/tokenize_text.py`。

如果想先用更小的真实数据跑一轮，只下载第一个条目分片：

```bash
bash scripts/prepare_zhwiki_small.sh data/zhwiki_small
uv run python scripts/build_char_tokenizer.py --input data/zhwiki_small/zhwiki_small.txt --out data/zhwiki_small/tokenizer.json --vocab-size 8000 --max-lines 2000
uv run python scripts/tokenize_text.py --input data/zhwiki_small/zhwiki_small.txt --tokenizer data/zhwiki_small/tokenizer.json --out-dir data/zhwiki_small/bin --val-ratio 0.02 --max-lines 2000
uv run python -m faof.train --config configs/wiki_smoke.json
```

## MVP 消融实验

用同一基础 checkpoint、以下列损失权重分别训练：

| 实验 | `anchor_weight` | `order_free_weight` | `bridge_weight` |
| --- | ---: | ---: | ---: |
| A 基线 | 0.0 | 0.0 | 0.0 |
| B 锚点 | 0.2 | 0.0 | 0.0 |
| C 锚点+集合 | 0.2 | 0.1 | 0.0 |
| D 锚点+集合+桥接 | 0.2 | 0.1 | 0.2 |

成功标准：

- 锚点 `top5@8` 和 `top5@16` 优于基线探针。
- 顺序无关的未来 token 召回率优于仅锚点方案。
- 填充跨度上，桥接 token 准确率优于下一 token 续写。
- 验证集下一 token 损失退化不超过约 3%。

生成并运行一个玩具规模的消融实验：

```bash
uv run python scripts/make_ablation_configs.py \
  --base configs/smoke.json \
  --out-dir configs/ablation_smoke \
  --run-root runs/ablation_smoke \
  --max-steps 60

bash scripts/run_ablation.sh configs/ablation_smoke
```

## 清洗 WikiExtractor 输出

如果用 WikiExtractor 的 JSON 输出抽取维基 dump：

```bash
uv run python scripts/jsonl_text_extract.py \
  --input data/zhwiki/extracted \
  --out data/zhwiki/zhwiki.txt \
  --min-chars 80
```

然后可以构建一个小型混合语料：

```bash
uv run python scripts/mix_corpora.py \
  --source data/zhwiki/zhwiki.txt:0.7 \
  --source data/clue/clue.txt:0.2 \
  --source data/web/web_sample.txt:0.1 \
  --out data/mixed_zh_small.txt \
  --max-docs 100000
```

## 许可证

[MIT](LICENSE)
