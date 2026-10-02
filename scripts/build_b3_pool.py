"""实验 B3 的候选数据池: 从 CheXpert Plus 完整表里抽出队友打分模型没见过的「胸片 + 报告」, 转成与他相同格式的 JPG。

为什么不用队友子集里的 10,293 对训练数据: 他的对齐模型在自己的训练对上 R@1 约 17-28%, 验证集只有 1.9%,
训练对的分数主要反映「背没背过」。B3 要比较「按对齐分数挑的 CPT 数据」与「随机挑的」, 打分必须用模型没见过的数据。

规则与队友子集一致 (/workspace/rafi/chexpert_plus_subset/README.md、subset_summary.json):
  - 只要正位片 (frontal_lateral == Frontal), 报告 = section_findings + "\\n" + section_impression 去首尾空白, 空的不要;
  - 一个检查 (deid_patient_id, patient_report_date_order) 取第一张正位片;
  - 预处理 = 队友 src/data/dataset.py 的 load_dicom_rgb (Chambon 附录 A: 拉伸 → MONOCHROME1 反色 → 直方图均衡),
    灰度、全分辨率、JPEG 质量 95 (与他 scripts/cache_plus_chambon_jpg.py 的 _convert_one 相同), 文件名规则 chambon_cache_jpg_path。
另外两条:
  - 排除队友 15,000 对子集里出现过的全部病人 (4,965 人), 自然也排除了他的测试集;
  - 每个病人只取一个检查, 固定种子随机抽 N 个病人 (默认 10,000), 来源更分散。

只读使用队友的代码和 /home/share 的原始数据; 输出全部写在 /workspace/chunqian/data/chexpert_b3/ 下。
清单含病人编号和报告原文, 属受控数据: 留在这台机器上, 不进 git、不拷出、不贴到聊天里; 屏幕只打印汇总数字。

环境: 我们自己的 conda 环境 chunqian, 需与队友相同版本的 opencv-python-headless==5.0.0.93、pydicom==3.0.2。

用法 (user0 容器, 在 tmux 里):
  python scripts/build_b3_pool.py verify --n 20      # 先拿子集里 20 张 (训练部分) 试转, 与队友已转好的 JPG 逐像素对比
  python scripts/build_b3_pool.py sample --n 10000   # 抽样, 写清单
  python scripts/build_b3_pool.py convert            # 把清单里的 DICOM 转成 JPG (多进程, 已转过的跳过)
之后打分: python scripts/score_chexpert_pairs.py --pairs <清单> --image-cache-root <JPG 目录> --split b3pool
"""
import argparse
import csv
import json
import random
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

RADPAIR = Path("/workspace/rafi/radpair")
SUBSET = Path("/workspace/rafi/chexpert_plus_subset/manifest.csv")
SUBSET_CACHE = Path("/workspace/rafi/chexpert_plus_subset/images_chambon")
FULL_CSV = Path("/home/share/chexpert_plus/df_chexpert_plus_240401.csv")
OUT_ROOT = Path("/workspace/chunqian/data/chexpert_b3")
MANIFEST = OUT_ROOT / "b3_pool_manifest.csv"
CACHE = OUT_ROOT / "images_chambon"
FIELDS = ["deid_patient_id", "patient_report_date_order", "path_to_dcm", "path_to_image", "image_path",
          "report_text", "our_split"]

sys.path.insert(0, str(RADPAIR))


def report_text(row: dict) -> str:
    findings = (row.get("section_findings") or "").strip()
    impression = (row.get("section_impression") or "").strip()
    return (findings + "\n" + impression).strip()


def dcm_root_from_subset() -> Path:
    """从队友清单里 image_path (磁盘绝对路径) 与 path_to_dcm (相对路径) 的关系推出 DICOM 根目录, 并核对一致。"""
    roots = Counter()
    for r in csv.DictReader(open(SUBSET, encoding="utf-8")):
        rel = r["path_to_dcm"].strip().lstrip("/")
        img = r["image_path"].strip()
        if img.endswith(rel):
            roots[img[: len(img) - len(rel)]] += 1
    if len(roots) != 1:
        raise SystemExit(f"无法确定 DICOM 根目录: {dict(roots)}")
    return Path(next(iter(roots)))


def convert_one(job):
    """与队友 _convert_one 相同的处理, 只是输出到我们自己的目录。"""
    dcm_abs, jpg_abs = job
    try:
        import cv2
        import numpy as np
        from src.data.dataset import load_dicom_rgb
        jpg = Path(jpg_abs)
        if jpg.is_file() and jpg.stat().st_size > 0:
            return jpg_abs, None
        arr = np.asarray(load_dicom_rgb(Path(dcm_abs)).convert("L"))
        jpg.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(jpg), arr, [int(cv2.IMWRITE_JPEG_QUALITY), 95]):
            return jpg_abs, "cv2.imwrite failed"
        return jpg_abs, None
    except Exception as e:  # noqa: BLE001
        return jpg_abs, f"{type(e).__name__}: {e}"


def run_jobs(jobs, workers):
    fails, done = [], 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for fut in as_completed([ex.submit(convert_one, j) for j in jobs]):
            path, err = fut.result()
            done += 1
            if err:
                fails.append(err)
            if done % 500 == 0 or done == len(jobs):
                print(f"  转换 {done}/{len(jobs)}  失败 {len(fails)}", flush=True)
    return fails


def cmd_verify(args):
    import numpy as np
    from PIL import Image
    from src.data.dataset import chambon_cache_jpg_path
    rows = [r for r in csv.DictReader(open(SUBSET, encoding="utf-8")) if r["our_split"] == "train"]
    random.Random(0).shuffle(rows)
    rows = rows[: args.n]
    tmp = OUT_ROOT / "verify"
    jobs = [(r["image_path"], str(chambon_cache_jpg_path(tmp, r["path_to_dcm"]))) for r in rows]
    for _, j in jobs:  # 每次都重新转, 保证对比的是这次的结果
        Path(j).unlink(missing_ok=True)
    fails = run_jobs(jobs, min(args.workers, len(jobs)))
    if fails:
        raise SystemExit(f"试转失败: {fails[:3]}")
    same, maxdiff = 0, 0
    for r, (_, j) in zip(rows, jobs):
        a = np.asarray(Image.open(j), dtype=np.int16)
        b = np.asarray(Image.open(chambon_cache_jpg_path(SUBSET_CACHE, r["path_to_dcm"])).convert("L"), dtype=np.int16)
        if a.shape != b.shape:
            raise SystemExit(f"尺寸不同: {a.shape} vs {b.shape}")
        d = int(np.abs(a - b).max())
        same += d == 0
        maxdiff = max(maxdiff, d)
    print(json.dumps({"n": len(rows), "identical": same, "max_abs_pixel_diff": maxdiff}, indent=2))
    print("全部逐像素相同 → 处理方式与队友一致" if same == len(rows) else "有差异, 先别转新数据, 把这几行贴出来")


def cmd_sample(args):
    excluded = {r["deid_patient_id"] for r in csv.DictReader(open(SUBSET, encoding="utf-8"))}
    root = dcm_root_from_subset()
    studies = {}  # (patient, order) -> 第一张正位片
    stats = Counter()
    with open(FULL_CSV, newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            stats["rows"] += 1
            if (row.get("frontal_lateral") or "").strip() != "Frontal":
                stats["skip_not_frontal"] += 1
                continue
            text = report_text(row)
            if not text:
                stats["skip_empty_text"] += 1
                continue
            pid = (row.get("deid_patient_id") or "").strip()
            if pid in excluded:
                stats["skip_in_teammate_subset"] += 1
                continue
            key = (pid, (row.get("patient_report_date_order") or "").strip())
            if key in studies:
                continue
            rel = (row.get("path_to_dcm") or "").strip().lstrip("/")
            studies[key] = {"deid_patient_id": pid, "patient_report_date_order": key[1], "path_to_dcm": rel,
                            "path_to_image": (row.get("path_to_image") or "").strip(),
                            "image_path": str(root / rel), "report_text": text, "our_split": "b3pool"}
    by_patient = {}
    for (pid, _), s in sorted(studies.items()):
        by_patient.setdefault(pid, []).append(s)
    rng = random.Random(args.seed)
    pids = sorted(by_patient)
    rng.shuffle(pids)
    chosen = [rng.choice(by_patient[p]) for p in pids[: args.n]]
    missing = sum(not Path(c["image_path"]).is_file() for c in chosen)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    with open(MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(chosen)
    summary = {**stats, "excluded_patients": len(excluded), "candidate_studies": len(studies),
               "candidate_patients": len(by_patient), "chosen": len(chosen), "seed": args.seed,
               "missing_dicom_files": missing, "dicom_root": str(root)}
    (OUT_ROOT / "b3_pool_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"清单 (含病人信息, 只留在这台机器): {MANIFEST}")


def cmd_convert(args):
    from src.data.dataset import chambon_cache_jpg_path
    rows = list(csv.DictReader(open(MANIFEST, encoding="utf-8")))
    jobs = [(r["image_path"], str(chambon_cache_jpg_path(CACHE, r["path_to_dcm"]))) for r in rows]
    fails = run_jobs(jobs, args.workers)
    rep = {"n": len(jobs), "fail": len(fails), "jpeg_quality": 95, "dicom_preprocess": "chambon_appendix_a",
           "cache_root": str(CACHE), "fails": fails[:20]}
    (OUT_ROOT / "b3_convert_report.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("n", "fail", "cache_root")}, indent=2))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify"); v.add_argument("--n", type=int, default=20); v.add_argument("--workers", type=int, default=8)
    s = sub.add_parser("sample"); s.add_argument("--n", type=int, default=10000); s.add_argument("--seed", type=int, default=42)
    c = sub.add_parser("convert"); c.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    {"verify": cmd_verify, "sample": cmd_sample, "convert": cmd_convert}[args.cmd](args)


if __name__ == "__main__":
    main()
