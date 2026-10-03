"""实验 B3: 按队友对齐模型的分数筛选 CheXpert Plus 图文对, 做成与 B2 (IU X-Ray) 同格式的图文 CPT 数据。

两组, 只差「怎么挑」:
  b3_cpt_top  : 10,000 对候选池 (scripts/build_b3_pool.py) 里对齐分数最高的 5,000 对
  b3_cpt_rand : 同一池子里固定种子随机挑的 5,000 对
候选池全是打分模型没见过的病人 (排除了队友子集的 4,965 人), 分数不是「背答案」的结果 (分布与验证集相同)。

样本格式与 data/convert_iu_xray.py 完全一致 (LLaVA-Med 第一阶段的 caption 式对齐, LLaMA-Factory 里 stage=sft):
  user "<image>" + 三种写报告提问之一 (固定种子) → assistant "Findings: ...\\nImpression: ..." (只放非空的段, 空白合并)。
报告分段从完整表按 path_to_dcm 取 (队友规则: 同一张片子取第一次出现的那行)。

图片: 候选池的 JPG 是 2828×2320 全分辨率 (与队友缓存一致), 训练时 Qwen 处理器还会缩到 ≤ 262,144 像素;
为免每步解码 650 万像素的大图, 先按长边 1024 用 LANCZOS 缩一份 (两组用同一份, 不影响比较)。

只读队友代码与 /home/share 原始数据; 输出: 训练 JSON 写到 data/processed/ (git 忽略, 含报告原文, 只留在服务器),
缩小的图写到 /workspace/chunqian/data/chexpert_b3/images_1024/, 不含病人信息的汇总写到 runs/b3_scores/。

用法 (user0 容器, chunqian 环境):
  python data/convert_chexpert_b3.py            # 默认各 5,000 对
"""
import argparse
import csv
import json
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "data" / "processed"
B3 = Path("/workspace/chunqian/data/chexpert_b3")
MANIFEST = B3 / "b3_pool_manifest.csv"
CACHE = B3 / "images_chambon"
SMALL = B3 / "images_1024"
SCORES = Path("/workspace/chunqian/runs/b3_scores/b3pool_scores.csv")
FULL_CSV = Path("/home/share/chexpert_plus/df_chexpert_plus_240401.csv")
SUMMARY = Path("/workspace/chunqian/runs/b3_scores/b3_sets_summary.json")
CPT_PROMPTS = [  # 与 data/convert_iu_xray.py 相同
    "Write the radiology report for this chest X-ray.",
    "Describe the findings and impression of this chest radiograph.",
    "Provide the findings and impression for this chest X-ray image.",
]


def clean(t: str) -> str:
    t = re.sub(r"\s+", " ", t or "").strip()
    return re.sub(r"\s+([.,;:])", r"\1", t)


def shrink(job):
    src, dst, long_side = job
    try:
        from PIL import Image
        d = Path(dst)
        if d.is_file() and d.stat().st_size > 0:
            return None
        im = Image.open(src).convert("L")
        w, h = im.size
        s = long_side / max(w, h)
        if s < 1:
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        d.parent.mkdir(parents=True, exist_ok=True)
        im.save(d, quality=95)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{src}: {type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    pool = {r["path_to_dcm"]: r for r in csv.DictReader(open(MANIFEST, encoding="utf-8"))}
    scores = {r["path_to_dcm"]: float(r["score"]) for r in csv.DictReader(open(SCORES, encoding="utf-8"))}
    keys = sorted(pool)
    if set(keys) != set(scores):
        raise SystemExit(f"清单 {len(keys)} 与分数 {len(scores)} 对不上")
    top = sorted(keys, key=lambda k: -scores[k])[: args.n]
    rand = random.Random(args.seed).sample(keys, args.n)

    need = set(top) | set(rand)
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

    small = {k: SMALL / (k[:-4] + ".jpg" if k.lower().endswith(".dcm") else k + ".jpg") for k in need}
    jobs = [(str(CACHE / (k[:-4] + ".jpg" if k.lower().endswith(".dcm") else k + ".jpg")), str(small[k]), args.long_side)
            for k in sorted(need)]
    errs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, fut in enumerate(as_completed([ex.submit(shrink, j) for j in jobs]), 1):
            if (e := fut.result()):
                errs.append(e)
            if i % 1000 == 0 or i == len(jobs):
                print(f"  缩图 {i}/{len(jobs)}  失败 {len(errs)}", flush=True)
    if errs:
        raise SystemExit(f"缩图失败 {len(errs)} 张, 例如 {errs[0]}")

    def build(sel, seed):
        rng = random.Random(seed)
        out = []
        for k in sel:
            fi, im = sections[k]
            report = ("Findings: " + fi) if fi else ""
            if im:
                report += ("\n" if report else "") + "Impression: " + im
            out.append({"messages": [{"role": "user", "content": "<image>" + rng.choice(CPT_PROMPTS)},
                                     {"role": "assistant", "content": report}],
                        "images": [str(small[k])]})
        return out

    info_fn = OUT / "dataset_info.json"
    info = json.loads(info_fn.read_text(encoding="utf-8")) if info_fn.exists() else {}
    tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant"}
    for name, sel in (("b3_cpt_top", top), ("b3_cpt_rand", rand)):
        (OUT / f"{name}.json").write_text(json.dumps(build(sel, args.seed), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        info[name] = {"file_name": f"{name}.json", "formatting": "sharegpt",
                      "columns": {"messages": "messages", "images": "images"}, "tags": tags}
    info_fn.write_text(json.dumps(info, indent=2), encoding="utf-8")

    def stats(sel):
        v = sorted(scores[k] for k in sel)
        n = len(v)
        return {"n": n, "mean": round(sum(v) / n, 4), "min": round(v[0], 4), "median": round(v[n // 2], 4),
                "max": round(v[-1], 4),
                "findings_present_pct": round(100 * sum(bool(sections[k][0]) for k in sel) / n, 1),
                "mean_report_words": round(sum(len((sections[k][0] + " " + sections[k][1]).split()) for k in sel) / n, 1)}

    summary = {"pool": len(keys), "top": stats(top), "rand": stats(rand), "overlap_top_rand": len(set(top) & set(rand)),
               "seed": args.seed, "image_long_side": args.long_side}
    SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("已写 data/processed/b3_cpt_top.json 与 b3_cpt_rand.json (含报告原文, 只留在服务器), 并登记进 dataset_info.json")


if __name__ == "__main__":
    main()
