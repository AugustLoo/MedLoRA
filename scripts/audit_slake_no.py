"""审计 SLAKE 短答训练集里的「否」: 有多少是在说「没有这个异常」。

用法: python scripts/audit_slake_no.py [train.json 路径] [--out results/xxx.json]
只读英文题 (训练用的就是英文部分)。SLAKE 没有放射报告, 所以这里量的是「否」的构成,
不是「报告没提」的比例; 见 results/README.md 对应一节的读法。
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NORMAL_Q = re.compile(r"\b(healthy|normal)\b", re.I)      # 「abnormal」里的 normal 前面没有词边界, 不会误中
ABNORMAL_Q = re.compile(r"abnormal|unhealthy", re.I)
DISEASE_Q = re.compile(r"Can (.+?) be observed|have (.+?)\?|evidence of (?:an? )?(.+?)\?|Is there an? (.+?)\?", re.I)


def classify(row: dict) -> str:
    """not_yn / organ / other / pos_global / neg_global / pos_specific / neg_specific。

    pos = 答案在说「有异常」, neg = 在说「没有异常」; global = 只问正常不正常,
    specific = 题目点了病名 (如 Can pneumothorax be observed ...)。
    """
    a = str(row["answer"]).strip().lower()
    if a not in ("yes", "no"):
        return "not_yn"
    ct = row.get("content_type")
    if ct == "Organ":
        return "organ"
    if ct != "Abnormality":
        return "other"
    q = row["question"]
    if NORMAL_Q.search(q):
        return "neg_global" if a == "yes" else "pos_global"
    kind = "global" if ABNORMAL_Q.search(q) else "specific"
    return ("pos_" if a == "yes" else "neg_") + kind


def summarize(rows: list[dict]) -> dict:
    kinds = [classify(r) for r in rows]
    no = [k for r, k in zip(rows, kinds) if str(r["answer"]).strip().lower() == "no"]
    yes = [k for r, k in zip(rows, kinds) if str(r["answer"]).strip().lower() == "yes"]
    xray = collections.Counter(k for r, k in zip(rows, kinds) if r.get("modality") == "X-Ray" and k != "not_yn")
    c = collections.Counter(kinds)
    by_img = collections.defaultdict(list)
    for r, k in zip(rows, kinds):
        by_img[r.get("img_id")].append((r, k))
    xray_imgs = collections.Counter()          # 每张 X 光片: 有题说它异常 / 只有题说它正常 / 没有这类题
    elsewhere = 0                              # 点名病种答否, 但同一张图别的题说有这个病 (只是不在问的位置)
    for pairs in by_img.values():
        ks = {k for _, k in pairs}
        if pairs[0][0].get("modality") == "X-Ray":
            xray_imgs["abnormal" if ks & {"pos_global", "pos_specific"} else "normal" if "neg_global" in ks else "no_finding_question"] += 1
        listed = " ; ".join(str(r["answer"]).lower() for r, k in pairs if k == "not_yn" and r.get("content_type") == "Abnormality")
        for r, k in pairs:
            m = DISEASE_Q.search(r["question"]) if k == "neg_specific" else None
            name = next((g for g in m.groups() if g), "").lower() if m else ""
            elsewhere += bool(name) and name in listed
    return {
        "samples": len(rows),
        "images": len({r.get("img_id") for r in rows}),
        "answer_no": len(no),
        "answer_yes": len(yes),
        "no_by_kind": dict(collections.Counter(no)),
        "yes_by_kind": dict(collections.Counter(yes)),
        "negative_findings": c["neg_global"] + c["neg_specific"],     # 在教「没有异常」的题
        "positive_findings": c["pos_global"] + c["pos_specific"],
        "xray": dict(xray),
        "xray_images": dict(xray_imgs),
        "neg_specific_but_disease_elsewhere_in_image": elsewhere,
        "neg_specific_questions": dict(collections.Counter(
            r["question"].strip() for r, k in zip(rows, kinds) if k == "neg_specific").most_common()),
        "specific_questions_seen": sorted({r["question"].strip() for r, k in zip(rows, kinds) if k.endswith("specific")}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(REPO / "data/raw/SLAKE/train.json"))
    ap.add_argument("--out")
    args = ap.parse_args()
    rows = [r for r in json.load(open(args.path, encoding="utf-8")) if str(r.get("q_lang", "en")).lower() == "en"]
    s = summarize(rows)
    n = s["samples"]
    print(f"样本 {n} 条, 图片 {s['images']} 张; 答 no {s['answer_no']} ({s['answer_no']/n:.1%}), 答 yes {s['answer_yes']}")
    print("「否」的构成:", s["no_by_kind"])
    print("「是」的构成:", s["yes_by_kind"])
    print(f"在教「没有异常」的题 {s['negative_findings']} 条 ({s['negative_findings']/n:.1%}); "
          f"在教「有异常」的题 {s['positive_findings']} 条 ({s['positive_findings']/n:.1%})")
    print("其中 X 光:", s["xray"])
    print("X 光片按图:", s["xray_images"])
    print(f"点名病种且答否的题 {sum(s['neg_specific_questions'].values())} 条, "
          f"其中同一张图别处有这个病的 {s['neg_specific_but_disease_elsewhere_in_image']} 条")
    if args.out:
        Path(args.out).write_text(json.dumps(s, ensure_ascii=False, indent=1), encoding="utf-8")
        print("已写", args.out)


if __name__ == "__main__":
    main()
