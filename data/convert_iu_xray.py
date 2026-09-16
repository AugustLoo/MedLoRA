"""IU X-Ray (Indiana University chest X-rays, OpenI) → 图文 CPT 语料 + 报告问答题。

数据: Kaggle raddar/chest-xrays-indiana-university, 目录里有
    indiana_reports.csv     uid, MeSH, Problems, image, indication, comparison, findings, impression
    indiana_projections.csv uid, filename, projection (Frontal / Lateral)
    images/images_normalized/*.png

输出 (data/processed/):
    iu_cpt.json  图文 CPT: user "<image>写报告" → assistant "Findings: ... Impression: ..."  (LLaMA-Factory sharegpt)
    iu_qa.json   报告问答 SFT: 由 Problems 字段造 yes/no 与 "有哪些异常" 题
    iu_split.json  按 uid 的 train/val 切分 (固定 seed), 供两边共用
并把 iu_cpt / iu_qa 登记进 dataset_info.json。

说明: 这里的「图文 CPT」采用 LLaVA-Med 第一阶段的做法, 即 caption 式对齐 (只对报告文本算 loss),
在 LLaMA-Factory 中以 stage=sft 实现; 与纯文本 stage=pt 的 CPT 区别在于输入含图像。
"""
import argparse
import csv
import glob
import json
import random
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed"

CPT_PROMPTS = [
    "Write the radiology report for this chest X-ray.",
    "Describe the findings and impression of this chest radiograph.",
    "Provide the findings and impression for this chest X-ray image.",
]
NORMAL_PAT = re.compile(r"^\s*normal\s*$", re.I)


def clean(t: str) -> str:
    t = (t or "").replace("XXXX", "").replace("xxxx", "")
    t = re.sub(r"\s+", " ", t).strip()
    t = re.sub(r"\s+([.,;:])", r"\1", t)
    return t


def find_file(root: Path, name: str) -> Path:
    hits = glob.glob(str(root / "**" / name), recursive=True)
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return Path(hits[0])


def problems_list(p: str) -> list[str]:
    p = (p or "").strip()
    if not p or NORMAL_PAT.match(p):
        return []
    return [x.strip() for x in p.split(";") if x.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/kaggle/input/chest-xrays-indiana-university")
    ap.add_argument("--max", type=int, default=5000, help="最多取多少份报告 (按 uid)")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frontal-only", action="store_true", default=True)
    args = ap.parse_args()

    root = Path(args.root)
    rep_fn = find_file(root, "indiana_reports.csv")
    proj_fn = find_file(root, "indiana_projections.csv")
    img_dir = find_file(root, "images_normalized")

    reports = {}
    with open(rep_fn, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            reports[str(r["uid"])] = r
    imgs = defaultdict(list)
    with open(proj_fn, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if args.frontal_only and str(r.get("projection", "")).lower() != "frontal":
                continue
            imgs[str(r["uid"])].append(r["filename"])

    uids = []
    for uid, r in reports.items():
        if uid not in imgs:
            continue
        if not clean(r.get("findings")) and not clean(r.get("impression")):
            continue
        uids.append(uid)
    uids.sort()
    rng = random.Random(args.seed)
    rng.shuffle(uids)
    uids = uids[: args.max]
    n_val = max(1, int(len(uids) * args.val_ratio))
    split = {"val": sorted(uids[:n_val]), "train": sorted(uids[n_val:])}

    cpt, qa = {"train": [], "val": []}, {"train": [], "val": []}
    for part, ids in split.items():
        for uid in ids:
            r = reports[uid]
            img = str(img_dir / imgs[uid][0])
            findings, impression = clean(r.get("findings")), clean(r.get("impression"))
            report = ""
            if findings:
                report += "Findings: " + findings
            if impression:
                report += ("\n" if report else "") + "Impression: " + impression
            cpt[part].append({
                "messages": [
                    {"role": "user", "content": "<image>" + rng.choice(CPT_PROMPTS)},
                    {"role": "assistant", "content": report},
                ],
                "images": [img],
            })
            probs = problems_list(r.get("Problems"))
            # 题 1: 有无异常
            qa[part].append({
                "messages": [
                    {"role": "user", "content": "<image>Are there any abnormal findings in this chest X-ray?\nAnswer with yes or no only."},
                    {"role": "assistant", "content": "Yes" if probs else "No"},
                ],
                "images": [img],
            })
            # 题 2: 异常列表 (仅异常片)
            if probs:
                qa[part].append({
                    "messages": [
                        {"role": "user", "content": "<image>What abnormalities are shown in this chest X-ray?\nAnswer with a short phrase."},
                        {"role": "assistant", "content": ", ".join(probs)},
                    ],
                    "images": [img],
                })

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "iu_split.json").write_text(json.dumps(split, indent=1), encoding="utf-8")
    info_fn = OUT / "dataset_info.json"
    info = json.loads(info_fn.read_text(encoding="utf-8")) if info_fn.exists() else {}
    tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
    for name, d in [("iu_cpt", cpt), ("iu_qa", qa)]:
        for part in ["train", "val"]:
            fn = OUT / f"{name}_{part}.json"
            fn.write_text(json.dumps(d[part], ensure_ascii=False, indent=1), encoding="utf-8")
            info[f"{name}_{part}"] = {"file_name": fn.name, "formatting": "sharegpt",
                                      "columns": {"messages": "messages", "images": "images"}, "tags": tags}
            print(f"{name}_{part}: {len(d[part])} -> {fn}")
    info_fn.write_text(json.dumps(info, indent=2), encoding="utf-8")
    n_abn = sum(1 for u in split["train"] if problems_list(reports[u].get("Problems")))
    print(f"reports used {len(uids)} (train {len(split['train'])}, val {len(split['val'])}); abnormal in train: {n_abn}")
    print("example:", json.dumps(cpt["train"][0], ensure_ascii=False)[:400])


if __name__ == "__main__":
    main()
