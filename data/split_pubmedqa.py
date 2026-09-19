"""把 PubMedQA pqa_labeled 的 1000 题按标签分层切成 train/test 两半, 切分固定入库。

为什么需要: 2026-09-17 的 CPT-only 评估显示 yes 偏置是 SLAKE 短答式 SFT 引入的, 不是 CPT
也不是基座 (见 results/README.md)。要验证「在 SFT 里混入三分类样本能否保住 maybe」,
就需要带真实 maybe 标签的训练样本, 而 pqa_artificial 几乎没有 maybe, 只能从 pqa_labeled 里划一半出来。

红线不变: 划给 train 的一半只用于训练, test 的一半只用于评估。
之前所有模型 (base / A / B1 / B2) 的 PubMedQA 分数都可以从已存的逐题预测里
重新在 test 半边上算出来, 不需要重跑, 见 scripts/eval_pubmedqa_split.py。

切分来源不依赖网络: 直接读 outputs/eval/pubmedqa_baseline_preds.jsonl 里的 pubid + gold。
输出 data/pubmedqa_split.json (入库, 永不再生成)。
用法: python data/split_pubmedqa.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "outputs" / "eval" / "pubmedqa_baseline_preds.jsonl"
OUT = REPO / "data" / "pubmedqa_split.json"
SEED = 42


def main():
    if OUT.exists():
        print(f"{OUT} 已存在, 不覆盖 (切分固定)。")
        return
    if not SRC.exists():
        raise SystemExit(f"需要 {SRC}; 先跑一次 eval/eval_pubmedqa.py --tag baseline")
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["gold"]].append(int(r["pubid"]))

    rng = random.Random(SEED)
    train, test = [], []
    for label in sorted(by_label):
        ids = sorted(by_label[label])
        rng.shuffle(ids)
        half = len(ids) // 2
        train += ids[:half]
        test += ids[half:]
    split = {
        "seed": SEED,
        "source": "PubMedQA pqa_labeled, 按 final_decision 分层对半切",
        "counts": {l: len(v) for l, v in sorted(by_label.items())},
        "train": sorted(train),
        "test": sorted(test),
    }
    OUT.write_text(json.dumps(split, indent=1), encoding="utf-8")
    print(f"总 {len(rows)} 题, 标签分布 {split['counts']}")
    print(f"train {len(train)} / test {len(test)} -> {OUT}")


if __name__ == "__main__":
    main()
