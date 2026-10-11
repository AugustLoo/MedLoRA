"""S2-CheXpertPlus 的训练数据: B3 候选池的全部 10,000 对「胸片 → 报告」, 只用 CheXpert Plus (老师 2026-10-08 的要求)。

与 B3 (data/convert_chexpert_b3.py) 用同一套做法, 直接调用它的函数, 保证一致:
  - 同一个候选池清单 (scripts/build_b3_pool.py, 排除了队友子集的全部病人, 所以与 CheXpert 测试 3,189 张不重叠);
  - 同一个缩图函数 shrink (长边 1024, LANCZOS, JPEG 质量 95, 已缩过的跳过 —— B3 已缩好的 7,519 张原样保留, 只补剩下的);
  - 同样的样本格式: user "<image>" + 三种写报告提问之一 (固定种子) → assistant "Findings: ...\\nImpression: ..." (只放非空的段)。
报告全部使用 (只有 Impression 的也用; 用户 2026-10-11 选 A)。

只读队友代码与 /home/share 原始数据; 输出 data/processed/cxp_s2_train.json (含报告原文, 不进 git, 只留在服务器),
不含病人信息的汇总写到 /workspace/chunqian/runs/cxp_s2_summary.json。
用法 (user0 容器, chunqian 环境): python data/convert_chexpert_s2.py
"""
import argparse
import csv
import json
import random
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from convert_chexpert_b3 import CACHE, CPT_PROMPTS, FULL_CSV, MANIFEST, OUT, SMALL, clean, shrink  # noqa: E402

SUMMARY = Path("/workspace/chunqian/runs/cxp_s2_summary.json")


def jpg_name(k: str) -> str:
    return k[:-4] + ".jpg" if k.lower().endswith(".dcm") else k + ".jpg"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    keys = sorted({r["path_to_dcm"] for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))})
    need = set(keys)
    sections = {}
    with open(FULL_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            rel = (row.get("path_to_dcm") or "").strip().lstrip("/")
            if rel in need and rel not in sections:
                sections[rel] = (clean(row.get("section_findings")), clean(row.get("section_impression")))
                if len(sections) == len(need):
                    break
    if len(sections) != len(need):
        raise SystemExit(f"完整表里只找到 {len(sections)} / {len(need)} 条报告")

    small = {k: SMALL / jpg_name(k) for k in keys}
    todo = [k for k in keys if not (small[k].is_file() and small[k].stat().st_size > 0)]
    print(f"候选池 {len(keys)} 对; 已缩好 {len(keys) - len(todo)}, 要补缩 {len(todo)}", flush=True)
    jobs = [(str(CACHE / jpg_name(k)), str(small[k]), args.long_side) for k in todo]
    errs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(shrink, j) for j in jobs]), 1):
            if (e := fut.result()):
                errs.append(e)
            if i % 500 == 0 or i == len(jobs):
                print(f"  缩图 {i}/{len(jobs)}  失败 {len(errs)}", flush=True)
    if errs:
        raise SystemExit(f"缩图失败 {len(errs)} 张, 例如 {errs[0]}")

    rng = random.Random(args.seed)
    rows, empty = [], 0
    for k in keys:
        fi, im = sections[k]
        report = ("Findings: " + fi) if fi else ""
        if im:
            report += ("\n" if report else "") + "Impression: " + im
        if not report:
            empty += 1
            continue
        rows.append({"messages": [{"role": "user", "content": "<image>" + rng.choice(CPT_PROMPTS)},
                                  {"role": "assistant", "content": report}],
                     "images": [str(small[k])]})
    (OUT / "cxp_s2_train.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")

    words = sorted(len(r["messages"][1]["content"].split()) for r in rows)
    summary = {"pool": len(keys), "used": len(rows), "empty_report_skipped": empty, "seed": args.seed,
               "image_long_side": args.long_side,
               "findings_present_pct": round(100 * sum(bool(sections[k][0]) for k in keys) / len(keys), 1),
               "report_words": {"mean": round(sum(words) / len(words), 1), "median": words[len(words) // 2],
                                "p90": words[int(len(words) * 0.9)], "max": words[-1]}}
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("已写 data/processed/cxp_s2_train.json (含报告原文, 只留在服务器)")


if __name__ == "__main__":
    main()
