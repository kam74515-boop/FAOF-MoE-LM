from __future__ import annotations

import argparse
import random
from pathlib import Path


ENTITIES = ["人工智能", "语言模型", "中文维基", "Transformer", "混合专家", "相对论", "量子计算", "长江"]
VERBS = ["提出", "描述", "连接", "影响", "包含", "改进", "解释", "形成"]
OBJECTS = ["未来锚点", "自然语序", "知识结构", "桥接片段", "内容集合", "功能词", "训练目标", "评估指标"]
ADVS = ["通常", "在现代研究中", "因此", "同时", "不过", "进一步", "从实验看", "在中文语境下"]


def make_doc() -> str:
    e1, e2 = random.sample(ENTITIES, 2)
    v1, v2 = random.sample(VERBS, 2)
    o1, o2 = random.sample(OBJECTS, 2)
    a1, a2 = random.sample(ADVS, 2)
    return (
        f"{e1}是一个重要概念，{a1}会{v1}{o1}。"
        f"{e2}与{e1}之间存在联系，并且能够{v2}{o2}。"
        f"在模型训练中，系统需要保持语义一致，也要避免把、被、的、了等功能词错位。"
        f"{a2}，未来锚点可以帮助模型提前规划内容，再通过桥接模块补全中间文本。"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--docs", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()
    random.seed(args.seed)
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for _ in range(args.docs):
            f.write(make_doc() + "\n")


if __name__ == "__main__":
    main()

