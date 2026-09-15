"""通用能力保持评估 (灾难性遗忘): TextVQA 验证集固定抽样。

同一批样本、同一随机种子, 在基座和每个 adapter 上各跑一次, 差值就是遗忘量。
数据: lmms-lab/textvqa (validation, 5000 条), 默认抽 300 条。

用法:
  python eval/eval_general.py --model Qwen/Qwen2.5-VL-3B-Instruct --tag baseline
  python eval/eval_general.py --model ... --adapter outputs/xxx --tag sft
"""
import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.metrics import vqa_accuracy  # noqa: E402
from medvlm.model import generate, load_model  # noqa: E402
from medvlm.prompts import GENERAL_VQA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=768 * 28 * 28)
    args = ap.parse_args()

    ds = load_dataset("lmms-lab/textvqa", split="validation")
    ds = ds.shuffle(seed=args.seed).select(range(min(args.n, len(ds))))
    model, processor = load_model(args.model, args.adapter, args.load_4bit, max_pixels=args.max_pixels)

    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores = []
    with open(out_dir / f"textvqa_{args.tag}_preds.jsonl", "w", encoding="utf-8") as fout:
        for r in tqdm(ds):
            pred = generate(model, processor, GENERAL_VQA.format(question=r["question"]), r["image"], 16)
            s = vqa_accuracy(pred, list(r["answers"]))
            scores.append(s)
            fout.write(json.dumps({"question_id": r.get("question_id"), "question": r["question"],
                                   "golds": list(r["answers"]), "pred": pred, "score": s}) + "\n")

    summary = {"model": args.model, "adapter": args.adapter, "n": len(scores), "seed": args.seed,
               "textvqa_acc": round(100 * sum(scores) / len(scores), 2)}
    (out_dir / f"textvqa_{args.tag}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
