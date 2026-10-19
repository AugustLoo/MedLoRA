"""三步训练的评测汇总: 原版 / S2-6datasets / 只做①(cpt1) / A / B / C 一张表, 存 results/interns2_3stage_eval_2026-10-19.json。

读 outputs/eval/ 的汇总 JSON (CheXpert Plus 的只有比例, 从容器带回); PubMedQA 只看考卷半边 (data/pubmedqa_split.json 的 test)。
用法: python scripts/summarize_3stage.py
"""
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
E = REPO / "outputs" / "eval"
HELD = set(json.loads((REPO / "data/pubmedqa_split.json").read_text(encoding="utf-8"))["test"])
MODELS = [("原版", "interns2_base"), ("S2-6datasets", "interns2_6ds"), ("只做①", "interns2_cpt1"),
          ("A (②)", "interns2_s2a"), ("B (①→②)", "interns2_s2b"), ("C (①→②→③)", "interns2_s3c")]


def j(name):
    p = E / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def pubmed_half(tag):
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
    return {"macro_f1": round(100 * sum(f1s) / 3, 2), "maybe_correct": sum(r["gold"] == r["pred"] == "maybe" for r in rows),
            "maybe_n": sum(r["gold"] == "maybe" for r in rows), "pred_maybe": sum(r["pred"] == "maybe" for r in rows)}


def get(d, *keys):
    for k in keys:
        if d is None:
            return None
        d = d.get(k)
    return d


out = {"date": "2026-10-19", "note": "35B 三步训练评测 (单次训练, 种子 42)。CheXpert Plus 为关键词判断; CheXbert 复核在容器排队。", "models": {}}
for name, tag in MODELS:
    s, n, z = j(f"slake_{tag}.json"), j(f"slake_noimg_{tag}.json"), j(f"slake_zh_{tag}.json")
    cx = j(f"chexpert_{tag}.json")
    out["models"][name] = {
        "tag": tag,
        "slake_closed": get(s, "metrics", "closed_acc"), "slake_open_em": get(s, "metrics", "open_em"),
        "slake_noimg_closed": get(n, "metrics", "closed_acc"), "slake_noimg_open_em": get(n, "metrics", "open_em"),
        "slake_zh_closed": get(z, "metrics", "closed_acc"), "slake_zh_open_em": get(z, "metrics", "open_em"),
        "slake_zh_open_f1": get(z, "metrics", "open_f1"),
        "cmexam": get(j(f"cmexam_{tag}.json"), "cmexam_acc"),
        "pubmedqa_heldout": pubmed_half(tag),
        "textvqa": get(j(f"textvqa_{tag}.json"), "textvqa_acc"), "mmbench": get(j(f"mmbench_{tag}.json"), "mmbench_acc"),
        "vqarad_closed": get(j(f"vqarad_{tag}.json"), "metrics", "closed_acc"), "vqarad_open_em": get(j(f"vqarad_{tag}.json"), "metrics", "open_em"),
        "pathvqa_closed": get(j(f"pathvqa_{tag}.json"), "metrics", "closed_acc"),
        "medqa": get(j(f"medqa_{tag}.json"), "medqa_acc"),
        "pneumonia_closed": get(j(f"pneumonia_{tag}.json"), "closed", "accuracy"),
        "pneumonia_desc_miss_pct": get(j(f"pneumonia_{tag}.json"), "freetext", "pneu_no_lung_abnormality_pct"),
        "cxr_desc_miss_pct": get(cx, "freetext", "abnormal_no_abnormality_reported_pct"),
        "cxr_contradiction_pct": get(cx, "freetext", "abnormal_closed_yes_but_desc_none_pct"),
        "cxr_closed_miss_pct": get(cx, "closed", "missed_abnormal_pct"),
        "cxr_false_alarm_pct": round(100 - cx["closed"]["specificity"], 1) if cx else None,
        "cxr_discrimination": get(cx, "freetext", "discrimination_pts"),
    }
# 原版 / S2-6datasets 的肺炎描述是 10-01 改判断规则前测的, 新口径的数字在 10-13 的快照里 (scripts/rescore_pneumonia.py 重算)
OLD = json.loads((REPO / "results/interns2_cxp_2026-10-13.json").read_text(encoding="utf-8"))["models"]
for name, key in (("原版", "base"), ("S2-6datasets", "S2-6datasets")):
    out["models"][name]["pneumonia_desc_miss_pct"] = OLD[key]["pneumonia_freetext"]["pneu_no_lung_abnormality_pct"]
(REPO / "results/interns2_3stage_eval_2026-10-19.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8", newline="\n")

ROWS = [("SLAKE 是非", "slake_closed"), ("SLAKE 开放 EM", "slake_open_em"), ("去掉图像 是非", "slake_noimg_closed"),
        ("去掉图像 开放", "slake_noimg_open_em"), ("SLAKE 中文 是非", "slake_zh_closed"), ("SLAKE 中文 开放", "slake_zh_open_em"),
        ("CMExam", "cmexam"), ("PubMedQA F1", ("pubmedqa_heldout", "macro_f1")), ("maybe 答对/55", ("pubmedqa_heldout", "maybe_correct")),
        ("TextVQA", "textvqa"), ("MMBench", "mmbench"), ("VQA-RAD 是非", "vqarad_closed"), ("PathVQA 是非", "pathvqa_closed"),
        ("MedQA", "medqa"), ("儿童肺炎 是非", "pneumonia_closed"), ("肺炎描述漏报%", "pneumonia_desc_miss_pct"),
        ("胸片描述漏报%", "cxr_desc_miss_pct"), ("胸片矛盾%", "cxr_contradiction_pct"), ("胸片是非漏报%", "cxr_closed_miss_pct"),
        ("胸片误报%", "cxr_false_alarm_pct"), ("胸片区分", "cxr_discrimination")]
names = [n for n, _ in MODELS]
print("| 测试 | " + " | ".join(names) + " |")
print("|---" * (len(names) + 1) + "|")
for label, key in ROWS:
    vals = []
    for n in names:
        m = out["models"][n]
        v = m.get(key[0], {}) and m[key[0]].get(key[1]) if isinstance(key, tuple) else m.get(key)
        vals.append("—" if v is None else f"{v}")
    print(f"| {label} | " + " | ".join(vals) + " |")
