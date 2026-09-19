"""用修正后的 CLOSED 评分重算 SLAKE 指标, 直接读逐题预测文件, 不用重跑模型。

背景: 旧评分把预测和标准答案都折成 yes/no/other 再比, 导致标准答案不是 yes/no 的
61 道题上, 任何非 yes/no 的回答都算对 (2026-09-19 发现)。

用法: python scripts/rescore_slake.py [--write]
  不带 --write 只打印对比; 带 --write 就地更新 outputs/eval/slake_*.json、
  逐题文件里的 score, 以及 results/*.json 里的 slake 段。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm import metrics as M  # noqa: E402

EV = REPO / "outputs" / "eval"
RES = REPO / "results"
# tag -> 写回哪个 results 文件 (None = 只更新 outputs/eval)
RESULT_FILES = {
    "baseline": ("baseline_2026-09-15.json", None),
    "sft_r16": ("sft_A_2026-09-15.json", None),
    "cpt_sft_r16": ("cpt_sft_B_2026-09-16.json", None),
    "cpt_iu_sft_r16": ("cpt_iu_sft_B2_2026-09-16.json", None),
    "cpt_only_b1": ("cpt_only_2026-09-17.json", "cpt_only_b1"),
    "cpt_only_b2": ("cpt_only_2026-09-17.json", "cpt_only_b2"),
}


def rescore(tag):
    fn = EV / f"slake_{tag}_preds.jsonl"
    if not fn.exists():
        return None
    rows = [json.loads(l) for l in open(fn, encoding="utf-8")]
    agg, by_mod = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["answer_type"].upper() == "CLOSED":
            s = M.closed_score(r["pred"], r["gold"])
            agg["closed_acc"].append(s)
            by_mod[r["modality"]]["closed_acc"].append(s)
        else:
            s = M.exact_match(r["pred"], r["gold"])
            rec = M.token_recall(r["pred"], r["gold"])
            agg["open_em"].append(s)
            agg["open_recall"].append(rec)
            agg["open_f1"].append(M.token_f1(r["pred"], r["gold"]))
            by_mod[r["modality"]]["open_recall"].append(rec)
        r["score"] = s
    mean = lambda xs: round(100 * sum(xs) / len(xs), 2) if xs else None
    return {
        "rows": rows,
        "metrics": {k: mean(v) for k, v in agg.items()},
        "by_modality": {m: {k: mean(v) for k, v in d.items()} for m, d in by_mod.items()},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    print(f"{'tag':<16}{'closed 旧':>10}{'closed 新':>10}{'差':>8}{'open EM':>10}{'recall':>9}")
    touched = {}
    for tag, (res_file, key) in RESULT_FILES.items():
        out = rescore(tag)
        if out is None:
            print(f"{tag:<16}  (缺预测文件, 跳过)")
            continue
        summ_fn = EV / f"slake_{tag}.json"
        old = json.loads(summ_fn.read_text(encoding="utf-8")) if summ_fn.exists() else {"metrics": {}}
        o = old["metrics"].get("closed_acc")
        n = out["metrics"]["closed_acc"]
        print(f"{tag:<16}{o if o is not None else '-':>10}{n:>10}{(n - o) if o is not None else 0:>+8.2f}"
              f"{out['metrics']['open_em']:>10}{out['metrics']['open_recall']:>9}")
        if not args.write:
            continue
        # 逐题文件
        with open(EV / f"slake_{tag}_preds.jsonl", "w", encoding="utf-8") as f:
            for r in out["rows"]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # 汇总文件
        old["metrics"] = out["metrics"]
        old["by_modality"] = out["by_modality"]
        old["scoring"] = "closed_score v2 (2026-09-19): 标准答案非 yes/no 的封闭题按字面比"
        summ_fn.write_text(json.dumps(old, indent=2, ensure_ascii=False), encoding="utf-8")
        # results/
        p = RES / res_file
        if p.exists():
            doc = touched.get(res_file) or json.loads(p.read_text(encoding="utf-8"))
            target = doc[key] if key else doc
            target["slake"]["metrics"] = out["metrics"]
            target["slake"]["by_modality"] = out["by_modality"]
            target["slake"]["scoring"] = old["scoring"]
            touched[res_file] = doc
    for fn, doc in touched.items():
        (RES / fn).write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.write:
        print("\n已写回 outputs/eval/ 与 results/")


if __name__ == "__main__":
    main()
