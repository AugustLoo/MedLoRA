"""中文医学知识测试: CMExam (中国执业医师考试) 测试集, 固定种子抽 1,000 道单选题, 纯文字 (2026-10-16 加, docs/PLAN_35B_CPT.md)。

与 MedQA 同一套做法: 只取回答里第一个选项字母 (A–E) 与标准答案比较, 不做选项轮换。
多选题 (答案不止一个字母) 不测; 抽样只用 test 划分, 训练数据已与整个 test / valid 划分隔离 (data/sft35_common.py)。

数据: 服务器上 /home/ubuntu/chunqian/data/sft35_raw/cmexam/test.json (scripts/dl_sft35.sh 从 fzkuji/CMExam 下载, Apache-2.0)。
用法 (远程模式同其他脚本: 设 MEDVLM_API_BASE):
  python eval/eval_cmexam.py --model <模型目录> --tag interns2_base
输出: outputs/eval/cmexam_<tag>.json 与 cmexam_<tag>_preds.jsonl
"""
import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

from tqdm import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "data"))
from medvlm.prompts import CMEXAM  # noqa: E402

DEFAULT_FILE = "/home/ubuntu/chunqian/data/sft35_raw/cmexam/test.json"
LETTERS = "ABCDE"


def load_items(path: str, n: int, seed: int) -> list[dict]:
    from sft35_common import load_any

    rows = []
    for i, r in enumerate(load_any(Path(path))):
        opts = {str(o["key"]).strip(): str(o["value"]).strip() for o in (r.get("Options") or []) if o.get("key")}
        ans = str(r.get("Answer", "")).strip().upper()
        if len(ans) == 1 and ans in opts and len(opts) >= 4:
            rows.append({"i": i, "question": str(r["Question"]).strip(), "options": dict(sorted(opts.items())), "gold": ans})
    random.Random(seed).shuffle(rows)
    return sorted(rows[:n] if n else rows, key=lambda r: r["i"])


def parse_letter(pred: str) -> str:
    m = re.search(r"(?<![A-Za-z])([A-E])(?![A-Za-z])", pred.upper())
    return m.group(1) if m else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--file", default=DEFAULT_FILE)
    ap.add_argument("--n", type=int, default=1000, help="抽多少道单选题 (0 = 全部单选题)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--load-4bit", action="store_true")
    args = ap.parse_args()

    from medvlm.model import backend_info, generate, load_model

    items = load_items(args.file, args.n, args.seed)
    model, processor = load_model(args.model, args.adapter, args.load_4bit)
    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores, unparsed, picked = [], 0, Counter()
    t0 = time.time()
    with open(out_dir / f"cmexam_{args.tag}_preds.jsonl", "w", encoding="utf-8") as fout:
        for r in tqdm(items):
            prompt = CMEXAM.format(question=r["question"], options="\n".join(f"{k}. {v}" for k, v in r["options"].items()))
            pred = generate(model, processor, prompt, None, 16)
            letter = parse_letter(pred)
            unparsed += letter == ""
            picked[letter or "?"] += 1
            s = float(letter == r["gold"])
            scores.append(s)
            fout.write(json.dumps({"i": r["i"], "gold": r["gold"], "pred_raw": pred, "pred": letter, "score": s},
                                  ensure_ascii=False) + "\n")
    summary = {"dataset": "CMExam test (单选, 固定种子抽样)", "file": args.file, "seed": args.seed, "model": args.model,
               "adapter": args.adapter, "backend": backend_info(model), "n": len(scores),
               "seconds": round(time.time() - t0, 1), "cmexam_acc": round(100 * sum(scores) / max(1, len(scores)), 2),
               "unparsed": unparsed, "pred_letters": dict(picked),
               "gold_letters": dict(Counter(r["gold"] for r in items))}
    (out_dir / f"cmexam_{args.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
