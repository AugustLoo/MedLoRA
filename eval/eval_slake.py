"""SLAKE 测试集评估: 医学 VQA 能力 (路线第 1 步的 zero-shot 基线, 以及之后每个 adapter)。

用法:
  # 云端 zero-shot 基线
  python eval/eval_slake.py --model Qwen/Qwen2.5-VL-3B-Instruct --tag baseline
  # 本机 4GB 显存冒烟测试 (只跑 10 条)
  python eval/eval_slake.py --model Qwen/Qwen2-VL-2B-Instruct --load-4bit --max-pixels 200704 --limit 10
  # 评估 LoRA adapter
  python eval/eval_slake.py --model Qwen/Qwen2.5-VL-3B-Instruct --adapter outputs/sft_slake_qlora_r16 --tag sft_r16

输出: outputs/eval/slake_<tag>.json (汇总指标) 和 slake_<tag>_preds.jsonl (逐条预测, 用于错例分析)
"""
import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm import metrics as M  # noqa: E402
from medvlm.model import generate, load_model  # noqa: E402
from medvlm.prompts import slake_prompt  # noqa: E402
from medvlm.slake import load_split, open_image  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条, 0=全部")
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--max-pixels", type=int, default=512 * 28 * 28)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    rows = load_split(args.split, lang="en")
    if args.limit:
        rows = rows[: args.limit]
    print(f"SLAKE {args.split} (en): {len(rows)} 条")

    model, processor = load_model(args.model, args.adapter, args.load_4bit, max_pixels=args.max_pixels)

    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_fn = out_dir / f"slake_{args.tag}_preds.jsonl"

    agg = defaultdict(list)
    by_modality = defaultdict(lambda: defaultdict(list))
    t0 = time.time()
    with open(pred_fn, "w", encoding="utf-8") as fout:
        for r in tqdm(rows):
            prompt = slake_prompt(r["question"], r["answer_type"])
            pred = generate(model, processor, prompt, open_image(r), args.max_new_tokens)
            closed = str(r["answer_type"]).upper() == "CLOSED"
            if closed:
                score = float(M.yes_no(pred) == M.yes_no(r["answer"]))
                agg["closed_acc"].append(score)
                by_modality[r["modality"]]["closed_acc"].append(score)
            else:
                em = M.exact_match(pred, r["answer"])
                rec = M.token_recall(pred, r["answer"])
                agg["open_em"].append(em)
                agg["open_recall"].append(rec)
                agg["open_f1"].append(M.token_f1(pred, r["answer"]))
                by_modality[r["modality"]]["open_recall"].append(rec)
                score = em
            fout.write(json.dumps({
                "qid": r["qid"], "modality": r["modality"], "content_type": r["content_type"],
                "answer_type": r["answer_type"], "question": r["question"],
                "gold": r["answer"], "pred": pred, "score": score,
            }, ensure_ascii=False) + "\n")

    def mean(xs):
        return round(100 * sum(xs) / len(xs), 2) if xs else None

    summary = {
        "model": args.model, "adapter": args.adapter, "split": args.split, "n": len(rows),
        "load_4bit": args.load_4bit, "max_pixels": args.max_pixels,
        "seconds": round(time.time() - t0, 1),
        "metrics": {k: mean(v) for k, v in agg.items()},
        "n_closed": len(agg["closed_acc"]), "n_open": len(agg["open_em"]),
        "by_modality": {m: {k: mean(v) for k, v in d.items()} for m, d in by_modality.items()},
    }
    (out_dir / f"slake_{args.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
