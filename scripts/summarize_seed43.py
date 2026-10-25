"""种子 43 重训的 A、B 评测汇总, 和种子 42 的 A、B 并排, 存 results/interns2_seed43_eval_2026-10-25.json。

字段与 summarize_3stage.py 一致 (同一套测试), 另加训练损失; 每个指标给两个种子的 B−A 差, 看「B 比 A」的方向在换种子后是否不变。
读 outputs/eval/ 的汇总 JSON (CheXpert Plus 只有比例, 从容器带回) 和 outputs/<tag>/train_log.jsonl。
用法: python scripts/summarize_seed43.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E = REPO / "outputs" / "eval"
HELD = set(json.loads((REPO / "data/pubmedqa_split.json").read_text(encoding="utf-8"))["test"])
TAGS = {"interns2_s2a": "A42", "interns2_s2b": "B42", "interns2_s2a_s43": "A43", "interns2_s2b_s43": "B43"}


def j(name):
    p = E / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


def pubmed_f1(tag):
    p = E / f"pubmedqa_{tag}_preds.jsonl"
    if not p.exists():
        return None
    rows = [r for r in map(json.loads, open(p, encoding="utf-8")) if r["pubid"] in HELD]
    f1s = []
    for c in ("yes", "no", "maybe"):
        tp = sum(r["gold"] == c == r["pred"] for r in rows)
        fp = sum(r["pred"] == c != r["gold"] for r in rows)
        fn = sum(r["gold"] == c != r["pred"] for r in rows)
        f1s.append(2 * tp / (2 * tp + fp + fn) if tp else 0.0)
    return round(100 * sum(f1s) / 3, 2)


def train_stats(tag):
    p = REPO / "outputs" / tag / "train_log.jsonl"
    if not p.exists():
        return {}
    rows = [r for r in map(json.loads, open(p, encoding="utf-8")) if "loss" in r and "eval" not in r]
    total = rows[-1]["total"]
    first = [r["loss"] for r in rows if r["step"] <= 0.1 * total]
    last = [r["loss"] for r in rows if r["step"] > 0.9 * total]
    return {"steps": rows[-1]["step"], "train_loss_first10pct_mean": round(sum(first) / len(first), 4),
            "train_loss_last10pct_mean": round(sum(last) / len(last), 4)}


def metrics(tag):
    s, n, z, cx = j(f"slake_{tag}.json"), j(f"slake_noimg_{tag}.json"), j(f"slake_zh_{tag}.json"), j(f"chexpert_{tag}.json")
    return {
        "slake_closed": get(s, "metrics", "closed_acc"), "slake_open_em": get(s, "metrics", "open_em"),
        "slake_noimg_closed": get(n, "metrics", "closed_acc"),
        "slake_zh_closed": get(z, "metrics", "closed_acc"), "slake_zh_open_em": get(z, "metrics", "open_em"),
        "cmexam": get(j(f"cmexam_{tag}.json"), "cmexam_acc"),
        "pubmedqa_heldout_f1": pubmed_f1(tag),
        "textvqa": get(j(f"textvqa_{tag}.json"), "textvqa_acc"), "mmbench": get(j(f"mmbench_{tag}.json"), "mmbench_acc"),
        "vqarad_closed": get(j(f"vqarad_{tag}.json"), "metrics", "closed_acc"),
        "pathvqa_closed": get(j(f"pathvqa_{tag}.json"), "metrics", "closed_acc"),
        "medqa": get(j(f"medqa_{tag}.json"), "medqa_acc"),
        "pneumonia_closed": get(j(f"pneumonia_{tag}.json"), "closed", "accuracy"),
        "pneumonia_desc_miss_pct": get(j(f"pneumonia_{tag}.json"), "freetext", "pneu_no_lung_abnormality_pct"),
        "cxr_desc_miss_pct": get(cx, "freetext", "abnormal_no_abnormality_reported_pct"),
        "cxr_closed_miss_pct": get(cx, "closed", "missed_abnormal_pct"),
        "cxr_false_alarm_pct": round(100 - cx["closed"]["specificity"], 1) if cx else None,
        **train_stats(tag),
    }


out = {"date": "2026-10-25",
       "note": "35B 模型 A、B 换种子 43 重训 (LoRA 初始值和数据顺序变, 数据不变; B 仍从种子 42 的第①步补丁出发)。"
               "CheXpert Plus 为关键词判断。b_minus_a: 每个种子内 B−A, 两个种子同号才算方向稳定。",
       "models": {tag: {"name": name, **metrics(tag)} for tag, name in TAGS.items()}}
M = out["models"]
keys = [k for k in M["interns2_s2a_s43"] if k not in ("name", "steps")]
out["b_minus_a"] = {}
for k in keys:
    d = {}
    for seed, (a, b) in (("42", ("interns2_s2a", "interns2_s2b")), ("43", ("interns2_s2a_s43", "interns2_s2b_s43"))):
        va, vb = M[a].get(k), M[b].get(k)
        d[seed] = None if va is None or vb is None else round(vb - va, 2)
    d["same_sign"] = None if None in (d["42"], d["43"]) else (d["42"] * d["43"] > 0)
    out["b_minus_a"][k] = d
(REPO / "results/interns2_seed43_eval_2026-10-25.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                                     encoding="utf-8", newline="\n")

print("| 指标 | " + " | ".join(TAGS.values()) + " | B−A (42) | B−A (43) | 同号 |")
print("|---" * 8 + "|")
for k in keys:
    d = out["b_minus_a"][k]
    print(f"| {k} | " + " | ".join("—" if M[t].get(k) is None else str(M[t][k]) for t in TAGS)
          + f" | {d['42']} | {d['43']} | {d['same_sign']} |")
