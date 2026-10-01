"""按 2026-10-01 验证过的口径, 从已有的 PneumoniaMNIST 逐题记录重算自由描述指标 (不用重跑模型)。

背景见 eval/eval_pneumonia.py 开头: 最初的「说成正常」判断 (says_normal) 漏掉 "no apparent abnormalities" 等说法,
且各模型漏掉的条数不同; 新口径看「去掉否定说法后有没有报告异常」, 分任何异常与肺部异常两种。
用法: python scripts/rescore_pneumonia.py interns2_base interns2_mix_0 ...   (默认: 35B 的五个模型)
输出: 打印对照表; 加 --json <路径> 另存全部指标。
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "eval"))
from eval_pneumonia import freetext_summary  # noqa: E402

DEFAULT = ["interns2_base", "interns2_mix_0", "interns2_abl_r0_s43", "interns2_mix_300", "interns2_abl_s43"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tags", nargs="*", default=DEFAULT)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()
    out = {}
    print(f"{'tag':24s} 肺炎片没报肺部异常  正常片没报肺部异常  区分(点)  肺炎片没报任何异常  是非答yes却没报  旧says_normal")
    for t in args.tags:
        rows = [json.loads(l) for l in open(REPO / "outputs" / "eval" / f"pneumonia_{t}_preds.jsonl", encoding="utf-8")]
        f = freetext_summary(rows)
        out[t] = f
        print(f"{t:24s} {f['pneu_no_lung_abnormality']:4d} ({f['pneu_no_lung_abnormality_pct']:5.1f}%)   "
              f"{f['normal_no_lung_abnormality']:4d} ({f['normal_no_lung_abnormality_pct']:5.1f}%)   "
              f"{f['lung_discrimination_pts']:6.1f}   {f['pneu_no_any_abnormality']:4d} ({f['pneu_no_any_abnormality_pct']:5.1f}%)   "
              f"{f['closed_yes_but_desc_no_lung_pct']:5.1f}%   {f['legacy_says_normal_pneu']}")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
