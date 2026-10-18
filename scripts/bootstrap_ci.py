"""关键指标的 95% 置信区间 (bootstrap), 以及模型之间的配对差: 判断两个模型的差别是不是测试集的抽样噪声。

读 outputs/eval/<测试>_<tag>_preds.jsonl 的逐题得分, 不调用模型, 不占显卡。第一个 tag 作对照, 其余每个和它按同一批题配对重抽样:
差值的区间不跨 0, 才算真有差别。

用法:
  python scripts/bootstrap_ci.py interns2_base interns2_mix_300 interns2_6ds
  python scripts/bootstrap_ci.py interns2_s2a interns2_s2b interns2_s3c --out results/ci_3stage.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
EVAL = REPO / "outputs" / "eval"


def _closed_open(rows, closed_of):
    return {"closed": {r["_k"]: r["score"] for r in rows if closed_of(r)},
            "open": {r["_k"]: r["score"] for r in rows if not closed_of(r)}}


# 测试 → (预测文件前缀, 每题的键, 怎么把逐题记录变成 {子项: {键: 0/1 得分}})
TESTS = {
    "SLAKE": ("slake", "qid", lambda rows: _closed_open(rows, lambda r: str(r["answer_type"]).upper() == "CLOSED")),
    "SLAKE 中文": ("slake_zh", "qid", lambda rows: _closed_open(rows, lambda r: str(r["answer_type"]).upper() == "CLOSED")),
    "SLAKE 去掉图像": ("slake_noimg", "qid", lambda rows: _closed_open(rows, lambda r: str(r["answer_type"]).upper() == "CLOSED")),
    "VQA-RAD": ("vqarad", "i", lambda rows: _closed_open(rows, lambda r: r["closed"])),
    "PathVQA": ("pathvqa", "i", lambda rows: _closed_open(rows, lambda r: r["closed"])),
    "MedQA": ("medqa", "i", lambda rows: {"acc": {r["_k"]: r["score"] for r in rows}}),
    "CMExam": ("cmexam", "i", lambda rows: {"acc": {r["_k"]: r["score"] for r in rows}}),
    "PneumoniaMNIST": ("pneumonia", "i", lambda rows: {"closed": {
        r["_k"]: float((r["closed"] == "yes") == (r["label"] == "pneumonia")) for r in rows}}),
}


def load(prefix, key, tag):
    p = EVAL / f"{prefix}_{tag}_preds.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(x) for x in open(p, encoding="utf-8")]
    for r in rows:
        r["_k"] = r[key]
    return rows


def ci(values, n_boot, rng):
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot))
    return 100 * means[int(0.025 * n_boot)], 100 * means[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("tags", nargs="+", help="第一个作对照")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="另存 JSON")
    args = ap.parse_args()
    ref = args.tags[0]
    out = {"reference": ref, "boot": args.boot, "seed": args.seed, "tests": {}}
    for test, (prefix, key, split) in TESTS.items():
        per = {t: load(prefix, key, t) for t in args.tags}
        if per[ref] is None:
            continue
        base = split(per[ref])
        for sub, ref_scores in base.items():
            name = f"{test} {sub}"
            res = {}
            for t in args.tags:
                if per[t] is None:
                    continue
                scores = split(per[t])[sub]
                common = sorted(set(scores) & set(ref_scores), key=str)
                if not common:
                    continue
                rng = random.Random(args.seed)
                v = [scores[k] for k in common]
                lo, hi = ci(v, args.boot, rng)
                row = {"n": len(common), "mean": round(100 * sum(v) / len(v), 2), "ci95": [round(lo, 2), round(hi, 2)]}
                if t != ref:
                    d = [scores[k] - ref_scores[k] for k in common]
                    dlo, dhi = ci(d, args.boot, random.Random(args.seed))
                    row["diff_vs_ref"] = round(100 * sum(d) / len(d), 2)
                    row["diff_ci95"] = [round(dlo, 2), round(dhi, 2)]
                    row["significant"] = dlo > 0 or dhi < 0
                res[t] = row
            out["tests"][name] = res
            print(f"\n{name}")
            for t, r in res.items():
                extra = ""
                if "diff_vs_ref" in r:
                    extra = (f"   差 {r['diff_vs_ref']:+.2f} [{r['diff_ci95'][0]:+.2f}, {r['diff_ci95'][1]:+.2f}]"
                             + ("  ← 有差别" if r["significant"] else "  (在噪声内)"))
                print(f"  {t:<22} {r['mean']:6.2f} [{r['ci95'][0]:6.2f}, {r['ci95'][1]:6.2f}]  n={r['n']}{extra}")
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print("\n已存", args.out)


if __name__ == "__main__":
    main()
