"""「漏报异常」正式指标: CheXpert Plus 测试集 (队友子集的 our_split=test, 3,189 张成人正位胸片), 只测不训。

方向 A (2026-09-30 老师同意)。SLAKE 只有 44 张异常图、PneumoniaMNIST 是儿童低清片, 结论都不稳;
这里每张片都有 CheXbert 从报告自动抽取的 14 类标签 (1 阳性 / 0 阴性 / -1 不确定 / 空 = 未提及)。
  异常片 = 任何一类阳性 (不算 Support Devices 与 No Finding); 正常片 = No Finding 阳性且没有其他阳性; 其余 (只有不确定) 单列。
每张片问两次:
  1. 是非题 "Are there any abnormal findings in this chest X-ray? Answer with yes or no only."
     → 异常片答 no = 漏报; 灵敏度 / 特异度; 按病变类别分别统计漏报率
  2. 自由描述 (与 PneumoniaMNIST 探针同一句) → 判断用 eval_pneumonia.py 里 2026-10-01 人工验证过的规则
     (去掉否定说法后有没有报告异常); 另看描述有没有提到该片标签里的具体病变 (按类别的关键词, 见 FINDING_WORDS)
     → 漏报 (异常片描述没报任何异常)、按病变的提及率、「是非题答 yes 却描述没报」的自相矛盾比例

运行位置: user0 容器 (数据只在这里, 只读队友目录); 模型在 5090 主机上起服务 (train/interns2/serve_for_container.sh),
本脚本经 http://172.17.0.1:23334/v1 调用 (同一台机器内部, 图片不落主机硬盘)。
CheXpert Plus 是受控数据: 逐题记录含由报告派生的标签, 只留在服务器 (outputs/eval/ 不进 git、不拷出);
汇总 json 只有比例与计数, 可以带回本机。

先缩图 (一次, 长边 1024, 写到 /workspace/chunqian/data/chexpert_test_1024/), 再并发请求接口。
用法 (user0 容器, chunqian 环境):
  export MEDVLM_API_BASE=http://172.17.0.1:23334/v1 MEDVLM_API_EXTRA='{"chat_template_kwargs":{"enable_thinking":false}}'
  python eval/eval_chexpert.py --model <主机上的模型目录> --tag interns2_base --concurrency 8
  python eval/eval_chexpert.py --model ... --tag test --limit 20      # 先试 20 张
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "eval"))
from medvlm import metrics as M  # noqa: E402
from medvlm.model import backend_info, generate, load_model  # noqa: E402
from medvlm.prompts import SLAKE_CLOSED  # noqa: E402
from eval_openended import NEGATION  # noqa: E402
from eval_pneumonia import Q_FREE, reports_any_abnormality, reports_lung_abnormality  # noqa: E402

PAIRS = Path("/workspace/rafi/chexpert_plus_subset/pairs_with_text_labels.csv")
CACHE = Path("/workspace/rafi/chexpert_plus_subset/images_chambon")
SMALL = Path("/workspace/chunqian/data/chexpert_test_1024")
Q_CLOSED = SLAKE_CLOSED.format(question="Are there any abnormal findings in this chest X-ray?")
FINDINGS = ["Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema", "Consolidation",
            "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture"]
FINDING_WORDS = {  # 描述里提到该类病变的关键词 (去否定后匹配)
    "Enlarged Cardiomediastinum": r"mediastin|cardiomediastinal",
    "Cardiomegaly": r"cardiomegal|enlarged (?:heart|cardiac)|(?:heart|cardiac silhouette)[^.;]*enlarg",
    "Lung Opacity": r"opacit|infiltrat|consolidat|haz",
    "Lung Lesion": r"nodul|mass|lesion|tumou?r",
    "Edema": r"edema|oedema|vascular congestion",
    "Consolidation": r"consolidat",
    "Pneumonia": r"pneumon|infection|infectious",
    "Atelectasis": r"atelecta",
    "Pneumothorax": r"pneumothora",
    "Pleural Effusion": r"effusion|fluid",
    "Pleural Other": r"pleural (?:thicken|scar)|thickening",
    "Fracture": r"fractur",
}


def label(v: str):
    v = (v or "").strip()
    return None if v == "" else float(v)


def shrink(job):
    src, dst = job
    from PIL import Image
    d = Path(dst)
    if d.is_file() and d.stat().st_size > 0:
        return None
    try:
        im = Image.open(src).convert("L")
        w, h = im.size
        s = 1024 / max(w, h)
        if s < 1:
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        d.parent.mkdir(parents=True, exist_ok=True)
        im.save(d, quality=95)
        return None
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--workers", type=int, default=16, help="缩图进程数")
    args = ap.parse_args()

    rows = [r for r in csv.DictReader(open(PAIRS, encoding="utf-8")) if r.get("our_split") == "test"]
    if args.limit:
        rows = rows[: args.limit]
    for r in rows:
        labs = {f: label(r.get(f)) for f in FINDINGS}
        pos = [f for f, v in labs.items() if v == 1.0]
        if pos:
            r["_group"] = "abnormal"
        elif label(r.get("No Finding")) == 1.0:
            r["_group"] = "normal"
        else:
            r["_group"] = "uncertain_only"
        r["_pos"] = pos
        rel = r["path_to_dcm"].strip().lstrip("/")
        rel = rel[:-4] + ".jpg" if rel.lower().endswith(".dcm") else rel + ".jpg"
        r["_src"], r["_img"] = str(CACHE / rel), str(SMALL / rel)
    groups = Counter(r["_group"] for r in rows)
    print(f"测试片 {len(rows)} 张: {dict(groups)}", flush=True)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        errs = [e for e in ex.map(shrink, [(r["_src"], r["_img"]) for r in rows], chunksize=16) if e]
    if errs:
        raise SystemExit(f"缩图失败 {len(errs)} 张: {errs[:2]}")
    print(f"缩图完成 {time.time() - t0:.0f}s", flush=True)

    model, processor = load_model(args.model, None, False)
    from PIL import Image

    def ask(r):
        img = Image.open(r["_img"]).convert("RGB")
        closed_raw = generate(model, processor, Q_CLOSED, img, 8)
        desc = generate(model, processor, Q_FREE, img, 128)
        return closed_raw, desc

    t0 = time.time()
    out_dir = REPO / "outputs" / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        for i, (closed_raw, desc) in enumerate(ex.map(ask, rows), 1):
            results.append((closed_raw, desc))
            if i % 200 == 0 or i == len(rows):
                print(f"  {i}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)

    pct = lambda a, b: round(100 * a / b, 2) if b else None  # noqa: E731
    g = defaultdict(Counter)
    per = {f: Counter() for f in FINDINGS}
    with open(out_dir / f"chexpert_{args.tag}_preds.jsonl", "w", encoding="utf-8") as fout:  # 只留服务器
        for i, (r, (closed_raw, desc)) in enumerate(zip(rows, results)):
            c = M.yes_no(closed_raw)
            rep_any, rep_lung = reports_any_abnormality(desc), reports_lung_abnormality(desc)
            body = NEGATION.sub(" ", desc)
            G = g[r["_group"]]
            G["n"] += 1
            G["closed_yes"] += c == "yes"
            G["closed_no"] += c == "no"
            G["closed_other"] += c == "other"
            G["desc_reports_any"] += rep_any
            G["desc_reports_lung"] += rep_lung
            G["closed_yes_desc_none"] += (c == "yes") and not rep_any
            for f in r["_pos"]:
                per[f]["n"] += 1
                per[f]["closed_yes"] += c == "yes"
                per[f]["desc_reports_any"] += rep_any
                per[f]["desc_mentions_finding"] += bool(re.search(FINDING_WORDS[f], body, re.I))
            fout.write(json.dumps({"i": i, "group": r["_group"], "pos": r["_pos"], "closed_raw": closed_raw,
                                   "closed": c, "desc": desc, "reports_any": rep_any, "reports_lung": rep_lung},
                                  ensure_ascii=False) + "\n")

    A, N = g["abnormal"], g["normal"]
    summary = {
        "dataset": "CheXpert Plus test split of the Topic 1 subset (our_split=test)", "model": args.model,
        "backend": backend_info(model), "n": len(rows), "groups": dict(groups), "seconds": round(time.time() - t0, 1),
        "closed": {"sensitivity": pct(A["closed_yes"], A["n"]), "specificity": pct(N["closed_no"], N["n"]),
                   "missed_abnormal": A["n"] - A["closed_yes"], "missed_abnormal_pct": pct(A["n"] - A["closed_yes"], A["n"]),
                   "unparsed": sum(x["closed_other"] for x in g.values()),
                   "uncertain_only_yes_pct": pct(g["uncertain_only"]["closed_yes"], g["uncertain_only"]["n"])},
        "freetext": {"abnormal_no_abnormality_reported_pct": pct(A["n"] - A["desc_reports_any"], A["n"]),
                     "abnormal_no_lung_abnormality_pct": pct(A["n"] - A["desc_reports_lung"], A["n"]),
                     "normal_no_abnormality_reported_pct": pct(N["n"] - N["desc_reports_any"], N["n"]),
                     "discrimination_pts": (round(pct(N["n"] - N["desc_reports_any"], N["n"])
                                                  - pct(A["n"] - A["desc_reports_any"], A["n"]), 2)
                                            if A["n"] and N["n"] else None),
                     "abnormal_closed_yes_but_desc_none_pct": pct(A["closed_yes_desc_none"], A["closed_yes"])},
        "per_finding": {f: {"n": c["n"], "closed_yes_pct": pct(c["closed_yes"], c["n"]),
                            "desc_reports_any_pct": pct(c["desc_reports_any"], c["n"]),
                            "desc_mentions_finding_pct": pct(c["desc_mentions_finding"], c["n"])}
                        for f, c in per.items() if c["n"]},
    }
    (out_dir / f"chexpert_{args.tag}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_finding"}, indent=2, ensure_ascii=False))
    print("逐题记录含由报告派生的标签, 只留在服务器; 汇总可带回本机:", out_dir / f"chexpert_{args.tag}.json")


if __name__ == "__main__":
    main()
