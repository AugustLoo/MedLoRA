"""外部医学知识测试集: MedQA (USMLE, 四选一, 纯文字; 2026-09-30 老师同意加入, 只测不训)。

为什么: SLAKE / PubMedQA 之外, 检验医学微调后纯文字的医学知识有没有丢 (或有没有涨)。
与 MMBench 同一套做法: 只取输出里第一个选项字母与标准答案比较, 不做选项轮换。

数据: HF GBaker/MedQA-USMLE-4-options, split test (1,273 题), 字段 question / options {A..D} / answer_idx /
meta_info (step1 或 step2&3)。首次运行需联网 (服务器上 HF_ENDPOINT=https://hf-mirror.com)。

用法:
  python eval/eval_medqa.py --model Qwen/Qwen2.5-VL-3B-Instruct --tag baseline_server
  python eval/eval_medqa.py --model ... --adapter outputs/xxx --tag sft_mix_300
远程模式同其他脚本: 设 MEDVLM_API_BASE (见 docs/API.md)。

输出: outputs/eval/medqa_<tag>.json 与 medqa_<tag>_preds.jsonl
"""
import argparse
import ast
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm.model import backend_info, generate, load_model  # noqa: E402
from medvlm.prompts import MEDQA  # noqa: E402

DATASET = "GBaker/MedQA-USMLE-4-options"
TEST_FILE = "phrases_no_exclude_test.jsonl"  # 只下测试文件 (2 MB), 不下 16 MB 的训练集
LETTERS = "ABCD"


def options_of(r) -> dict:
    o = r["options"]
    if isinstance(o, str):  # 少数导出把字典存成字符串
        o = ast.literal_eval(o)
    return {k: str(o[k]).strip() for k in LETTERS if k in o}


def parse_letter(pred: str) -> str:
    m = re.search(r"\b([ABCD])\b", pred.upper())
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-3B-Instruct")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    ds = load_dataset(DATASET, data_files={"test": TEST_FILE}, split="test", verification_mode="no_checks")
    if args.limit:
        ds = ds.select(range(args.limit))
    model, processor = load_model(args.model, args.adapter, args.load_4bit)

    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores, by_step, unparsed = [], defaultdict(list), 0
    t0 = time.time()
    with open(out_dir / f"medqa_{args.tag}_preds.jsonl", "w", encoding="utf-8") as fout:
        for i, r in enumerate(tqdm(ds)):
            opts = options_of(r)
            prompt = MEDQA.format(question=str(r["question"]).strip(),
                                  options="\n".join(f"{k}. {v}" for k, v in opts.items()))
            pred = generate(model, processor, prompt, None, 8)
            letter = parse_letter(pred)
            unparsed += letter == ""
            gold = str(r["answer_idx"]).strip().upper()
            s = float(letter == gold)
            scores.append(s)
            by_step[str(r.get("meta_info") or "unknown")].append(s)
            fout.write(json.dumps({"i": i, "step": r.get("meta_info"), "gold": gold, "pred_raw": pred,
                                   "pred": letter, "score": s}, ensure_ascii=False) + "\n")

    summary = {
        "dataset": f"{DATASET}/test", "model": args.model, "adapter": args.adapter, "backend": backend_info(model),
        "n": len(scores), "seconds": round(time.time() - t0, 1),
        "medqa_acc": round(100 * sum(scores) / len(scores), 2), "unparsed": unparsed,
        "by_step": {k: {"n": len(v), "acc": round(100 * sum(v) / len(v), 2)} for k, v in sorted(by_step.items())},
    }
    (out_dir / f"medqa_{args.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False),
                                                     encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
