"""把 PubMedQA pqa_labeled 的 train 半边转成 SFT 训练数据 (纯文本, 三分类), 用于实验 C1。

实验 C1 要回答的问题: yes 偏置是 SLAKE 短答式 SFT 引入的 (见 results/README.md 的 CPT-only 一节),
那么在 SFT 数据里混入三分类样本, maybe 能不能保住, SLAKE 会不会掉。

提示词与评估完全一致 (medvlm.prompts.PUBMEDQA), 答案就是 final_decision。
只用 data/pubmedqa_split.json 的 train 半边; test 半边永不进训练。

类别平衡: maybe 在 train 半边只有 ~55 条, 直接混进 4,919 条 SLAKE 里会被淹没。
默认每类抽 300 条 (不足则有放回重复), 共 900 条, 约占混合集的 15%。
--per-class 0 表示不平衡, 原样使用。

用法 (服务器上):
  python data/convert_pubmedqa_sft.py --per-class 300
输出: data/processed/pubmedqa_sft_train.json, 并登记进 dataset_info.json
"""
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.prompts import PUBMEDQA  # noqa: E402

OUT = REPO / "data" / "processed"
SPLIT = REPO / "data" / "pubmedqa_split.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=300, help="每个标签抽多少条; 0 = 不平衡")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not SPLIT.exists():
        raise SystemExit(f"缺 {SPLIT}; 它应随仓库入库, 不要在服务器上重新生成")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train_ids = set(split["train"])

    from datasets import load_dataset
    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")

    by_label = defaultdict(list)
    for r in ds:
        if int(r["pubid"]) not in train_ids:
            continue
        ctx = r["context"]
        ctx = "\n".join(ctx["contexts"]) if isinstance(ctx, dict) else str(ctx)
        by_label[r["final_decision"]].append({
            "messages": [
                {"role": "user", "content": PUBMEDQA.format(context=ctx.strip(), question=r["question"].strip())},
                {"role": "assistant", "content": r["final_decision"]},
            ]
        })
    got = {k: len(v) for k, v in sorted(by_label.items())}
    if sum(got.values()) != len(train_ids):
        print(f"警告: 命中 {sum(got.values())} 条, 切分里有 {len(train_ids)} 条", file=sys.stderr)

    rng = random.Random(args.seed)
    samples = []
    if args.per_class > 0:
        for label in sorted(by_label):
            pool = by_label[label]
            picked = [rng.choice(pool) for _ in range(args.per_class)] if len(pool) < args.per_class \
                else rng.sample(pool, args.per_class)
            samples += picked
    else:
        for label in sorted(by_label):
            samples += by_label[label]
    rng.shuffle(samples)

    OUT.mkdir(parents=True, exist_ok=True)
    fn = OUT / "pubmedqa_sft_train.json"
    fn.write_text(json.dumps(samples, ensure_ascii=False, indent=1), encoding="utf-8")

    info_fn = OUT / "dataset_info.json"
    info = json.loads(info_fn.read_text(encoding="utf-8")) if info_fn.exists() else {}
    info["pubmedqa_sft_train"] = {
        "file_name": fn.name,
        "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"},
    }
    info_fn.write_text(json.dumps(info, indent=2), encoding="utf-8")

    dist = Counter(s["messages"][1]["content"] for s in samples)
    print(f"train 半边可用: {got}")
    print(f"输出 {len(samples)} 条 -> {fn}; 标签分布 {dict(sorted(dist.items()))}")
    print("dataset_info 现有:", sorted(info))


if __name__ == "__main__":
    main()
