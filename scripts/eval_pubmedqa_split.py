"""在 PubMedQA 的 test 半边 (data/pubmedqa_split.json) 上重算所有模型的分数。

为什么: 实验 C1 用 train 半边训练, 所以它在全部 1000 题上的分数会被污染。
把每个模型的逐题预测按 test 半边过滤后重算, 就得到一把公平的尺子;
之前的模型不需要重跑, 它们的逐题预测都在 outputs/eval/。

用法: python scripts/eval_pubmedqa_split.py [--half test|train|all]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EV = REPO / "outputs" / "eval"
LABELS = ["yes", "no", "maybe"]
TAGS = [("基座", "baseline"), ("基座(服务器)", "baseline_server"),
        ("B1 只CPT", "cpt_only_b1"), ("B2 只CPT", "cpt_only_b2"),
        ("A 只SFT", "sft_r16"), ("B1 CPT+SFT", "cpt_sft_r16"), ("B2 CPT+SFT", "cpt_iu_sft_r16"),
        ("C1 SFT+回放", "sft_mix_pubmedqa_r16"),
        ("C2 回放100", "sft_mix_100"), ("C2 回放300", "sft_mix_300"), ("C2 回放300 s43", "sft_mix_300_s43"),
        ("C1 回放900 s43", "sft_mix_900_s43")]


def macro_f1(pairs):
    f1s = []
    for lab in LABELS:
        tp = sum(1 for g, p in pairs if g == lab and p == lab)
        fp = sum(1 for g, p in pairs if g != lab and p == lab)
        fn = sum(1 for g, p in pairs if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return 100 * sum(f1s) / len(f1s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="test", choices=["test", "train", "all"])
    args = ap.parse_args()

    split = json.loads((REPO / "data" / "pubmedqa_split.json").read_text(encoding="utf-8"))
    keep = None if args.half == "all" else set(split[args.half])

    print(f"半边: {args.half} ({'全部 1000' if keep is None else len(keep)} 题)")
    print(f"{'模型':<16}{'acc':>8}{'macroF1':>9}{'  预测 yes/no/maybe':<24}{'maybe 答对':>10}")
    gold_shown = False
    for name, tag in TAGS:
        fn = EV / f"pubmedqa_{tag}_preds.jsonl"
        if not fn.exists():
            continue
        rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
        if keep is not None:
            rows = [r for r in rows if int(r["pubid"]) in keep]
        if not rows:
            continue
        pairs = [(r["gold"], r["pred"]) for r in rows]
        acc = 100 * sum(g == p for g, p in pairs) / len(pairs)
        dist = Counter(p for _, p in pairs)
        nmaybe = sum(1 for g, p in pairs if g == "maybe" and p == "maybe")
        gmaybe = sum(1 for g, _ in pairs if g == "maybe")
        if not gold_shown:
            gd = Counter(g for g, _ in pairs)
            print(f"{'真实标签':<16}{'':>8}{'':>9}  {gd['yes']:>4}/{gd['no']:>4}/{gd['maybe']:<4}"
                  f"{'':<10}{gmaybe:>10}")
            gold_shown = True
        print(f"{name:<16}{acc:>8.2f}{macro_f1(pairs):>9.2f}  {dist['yes']:>4}/{dist['no']:>4}/{dist['maybe']:<4}"
              f"{'':<10}{nmaybe:>6}/{gmaybe}")


if __name__ == "__main__":
    main()
