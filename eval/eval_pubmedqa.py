"""PubMedQA pqa_labeled 评估: 文本医学推理 + 可靠性 (yes/no/maybe 三分类)。

作为幻觉/可靠性指标之一: 模型能否在证据不足时回答 maybe, 而不是硬答 yes。
报告 accuracy 和 macro-F1, 以及各类别的预测分布。

用法:
  python eval/eval_pubmedqa.py --model Qwen/Qwen2.5-VL-3B-Instruct --tag baseline
  python eval/eval_pubmedqa.py --model ... --adapter outputs/xxx --tag sft
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.metrics import normalize  # noqa: E402
from medvlm.model import generate, load_model  # noqa: E402
from medvlm.prompts import PUBMEDQA  # noqa: E402

LABELS = ["yes", "no", "maybe"]


def parse_label(pred: str) -> str:
    n = normalize(pred)
    for lab in LABELS:
        if n.startswith(lab):
            return lab
    for lab in LABELS:
        if lab in n.split():
            return lab
    return "maybe"  # 无法解析按 maybe 计, 并在 unparsed 里统计


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    if args.limit:
        ds = ds.select(range(args.limit))
    model, processor = load_model(args.model, args.adapter, args.load_4bit)

    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    golds, preds, unparsed = [], [], 0
    with open(out_dir / f"pubmedqa_{args.tag}_preds.jsonl", "w", encoding="utf-8") as fout:
        for r in tqdm(ds):
            ctx = "\n".join(r["context"]["contexts"])
            raw = generate(model, processor, PUBMEDQA.format(context=ctx, question=r["question"]), None, 8)
            lab = parse_label(raw)
            if not any(l in normalize(raw).split() for l in LABELS):
                unparsed += 1
            golds.append(r["final_decision"])
            preds.append(lab)
            fout.write(json.dumps({"pubid": r["pubid"], "gold": r["final_decision"], "pred": lab, "raw": raw}) + "\n")

    summary = {
        "model": args.model, "adapter": args.adapter, "n": len(golds),
        "accuracy": round(100 * accuracy_score(golds, preds), 2),
        "macro_f1": round(100 * f1_score(golds, preds, labels=LABELS, average="macro"), 2),
        "pred_dist": dict(Counter(preds)), "gold_dist": dict(Counter(golds)),
        "unparsed": unparsed,
    }
    (out_dir / f"pubmedqa_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
