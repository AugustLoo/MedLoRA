"""S2-6datasets 的四个新训练来源 (只用训练部分): VQA-RAD / PathVQA / MedQA / PneumoniaMNIST。

设计: docs/superpowers/specs/2026-10-08-s2-6datasets-design.md
提示词与评测逐字相同 (medvlm.prompts; PneumoniaMNIST 与 eval/eval_pneumonia.py 的是非题同一句),
输出与 slake_train.json 同格式 (messages + images), train/interns2/train_lora.py 直接读。

防泄漏 (自动, 统计写进 <out-dir>/sft6_stats.json):
  - 训练样本与测试部分「同图同题」(VQA) / 「同题干」(MedQA) / 「同图」(PneumoniaMNIST) 的剔除;
  - 训练部分内部完全重复的样本只留一条;
  - 统计训练 / 测试共用的图片数; VQA 类默认再去掉所有问「测试部分图片」的训练题
    (VQA-RAD 203 张测试图里 202 张也在训练部分, 不去掉的话考试前就见过几乎所有考卷上的图)。
抽样: 固定种子, 数量是参数 (以后加大改数字重跑)。只为抽中的样本存图, 文件名是图片哈希。

用法 (5090 主机, s2train 环境, 走 hf-mirror; 一般由 train/interns2/run_6ds.sh data 调用):
  export HF_HOME=/home/ubuntu/chunqian/hf HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
  python data/convert_sft6.py --out-dir /home/ubuntu/chunqian/data/processed --image-dir /home/ubuntu/chunqian/data/sft6_images
  默认 --pathvqa 5000 --medqa 3000 --pneumonia 2000, VQA-RAD 全部; 只做某几个: --only vqarad medqa
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from medvlm import metrics as M  # noqa: E402
from medvlm.prompts import MEDQA, SLAKE_CLOSED, slake_prompt  # noqa: E402

PNEUMONIA_Q = SLAKE_CLOSED.format(question="Does this chest X-ray show pneumonia?")  # 与 eval_pneumonia.Q_CLOSED 相同
EXPECTED_TRAIN = {"vqarad": 1793, "pathvqa": 19654, "medqa": 10178, "pneumonia": 4708}  # 公开说明里的训练部分大小
LETTERS = "ABCD"


def norm_q(text) -> str:
    return " ".join(str(text).lower().split())


def image_key(img) -> str:
    """按像素算的指纹: 同样的像素 (不论 L / RGB 模式) 得到同一个键。"""
    im = img.convert("RGB")
    h = hashlib.md5(f"{im.width}x{im.height}".encode())
    h.update(im.tobytes())
    return h.hexdigest()


def is_closed(answer) -> bool:
    """与 eval/eval_medvqa.py 相同: 金标归一化后是 yes / no 的算封闭题。"""
    return M.normalize(answer) in {"yes", "no"}


def vqa_row(question, answer, image_path) -> dict:
    kind = "CLOSED" if is_closed(answer) else "OPEN"
    return {"messages": [{"role": "user", "content": "<image>" + slake_prompt(str(question), kind)},
                         {"role": "assistant", "content": str(answer).strip()}],
            "images": [str(image_path)]}


def options_dict(options) -> dict:
    if isinstance(options, str):  # 少数导出把字典存成字符串 (与 eval_medqa.options_of 相同)
        options = ast.literal_eval(options)
    return {k: str(options[k]).strip() for k in LETTERS if k in options}


def medqa_row(question, options, answer_idx) -> dict:
    opts = options_dict(options)
    prompt = MEDQA.format(question=str(question).strip(),
                          options="\n".join(f"{k}. {v}" for k, v in opts.items()))
    return {"messages": [{"role": "user", "content": prompt},
                         {"role": "assistant", "content": str(answer_idx).strip().upper()}]}


def pneumonia_row(label, image_path) -> dict:
    return {"messages": [{"role": "user", "content": "<image>" + PNEUMONIA_Q},
                         {"role": "assistant", "content": "yes" if as_int(label) == 1 else "no"}],
            "images": [str(image_path)]}


def as_int(v) -> int:
    """MedMNIST 的标签有时是形如 [1] 的列表。"""
    while isinstance(v, (list, tuple)):
        v = v[0]
    return int(v)


def sample(items, n: int, seed: int) -> list:
    items = list(items)
    if n <= 0 or n >= len(items):
        return items
    return random.Random(seed).sample(items, n)


def dedupe(items, key):
    seen, kept = set(), []
    for it in items:
        k = key(it)
        if k not in seen:
            seen.add(k)
            kept.append(it)
    return kept, len(items) - len(kept)


def _save(image, image_dir: Path, key: str) -> Path:
    path = image_dir / f"{key}.png"
    if not path.is_file():
        image.convert("RGB").save(path)
    return path


def build_vqa(train, test, image_dir: Path, n: int, seed: int, exclude_test_images: bool = False):
    """VQA-RAD / PathVQA: 剔除与测试「同图同题」的样本, 去重, 抽 n 条 (0 = 全部), 只为抽中的存图。
    exclude_test_images: 再去掉所有问「测试部分图片」的训练题 (VQA-RAD 官方切分按问题切, 203 张测试图里 202 张
    也在训练部分; 2026-10-08 决定 S2-6datasets 只用测试部分没有的图, 让 VQA-RAD 仍是「没见过的图」)。"""
    test_imgs, test_pairs = set(), set()
    for j in range(len(test)):
        r = test[j]
        k = image_key(r["image"])
        test_imgs.add(k)
        test_pairs.add((k, norm_q(r["question"])))
    items = []
    for i in range(len(train)):
        r = train[i]
        k = image_key(r["image"])
        items.append((i, k, norm_q(r["question"]), M.normalize(r["answer"])))
    total = len(items)
    items, dup = dedupe(items, key=lambda t: t[1:])
    clean = [t for t in items if (t[1], t[2]) not in test_pairs]
    n_pair_dropped = len(items) - len(clean)
    n_img_dropped = 0
    if exclude_test_images:
        kept = [t for t in clean if t[1] not in test_imgs]
        n_img_dropped, clean = len(clean) - len(kept), kept
    picked = sample(clean, n, seed)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, k, _, _ in picked:
        r = train[i]
        rows.append(vqa_row(r["question"], r["answer"], _save(r["image"], image_dir, k)))
    train_imgs = {t[1] for t in items}
    stats = {"train_total": total, "duplicates_dropped": dup,
             "same_image_and_question_as_test_dropped": n_pair_dropped,
             "rows_on_test_images_dropped": n_img_dropped,
             "used": len(rows), "used_closed": sum(is_closed(r["messages"][1]["content"]) for r in rows),
             "train_images": len(train_imgs), "test_images": len(test_imgs),
             "images_shared_with_test": len(train_imgs & test_imgs),
             "used_rows_on_test_images": sum(t[1] in test_imgs for t in picked)}
    return rows, stats


def build_medqa(train, test, n: int, seed: int):
    test_q = {norm_q(test[j]["question"]) for j in range(len(test))}
    items = [(i, norm_q(train[i]["question"])) for i in range(len(train))]
    total = len(items)
    items, dup = dedupe(items, key=lambda t: t[1])
    clean = [t for t in items if t[1] not in test_q]
    picked = sample(clean, n, seed)
    rows = [medqa_row(train[i]["question"], train[i]["options"], train[i]["answer_idx"]) for i, _ in picked]
    stats = {"train_total": total, "duplicates_dropped": dup,
             "same_question_as_test_dropped": len(items) - len(clean), "used": len(rows)}
    return rows, stats


def build_pneumonia(train, test_images, image_dir: Path, n: int, seed: int, label_col: str = "label"):
    """肺炎 / 正常各抽 n // 2 张 (不够就全用, 不重复抽), 剔除与测试同图的。"""
    test_keys = {image_key(im) for im in test_images}
    items = [(i, image_key(train[i]["image"]), as_int(train[i][label_col])) for i in range(len(train))]
    total = len(items)
    items, dup = dedupe(items, key=lambda t: t[1])
    clean = [t for t in items if t[1] not in test_keys]
    by_label = defaultdict(list)
    for t in clean:
        by_label[t[2]].append(t)
    picked = []
    for lab in (1, 0):
        picked += sample(by_label[lab], n // 2 if n > 0 else 0, seed + lab)
    random.Random(seed).shuffle(picked)
    image_dir.mkdir(parents=True, exist_ok=True)
    rows = [pneumonia_row(lab, _save(train[i]["image"], image_dir, k)) for i, k, lab in picked]
    stats = {"train_total": total, "train_pneumonia": sum(t[2] == 1 for t in items),
             "train_normal": sum(t[2] == 0 for t in items), "duplicates_dropped": dup,
             "same_image_as_test_dropped": len(items) - len(clean), "used": len(rows),
             "used_pneumonia": sum(t[2] == 1 for t in picked), "used_normal": sum(t[2] == 0 for t in picked)}
    return rows, stats


# ---------------------------------------------------------------- 下载与写文件 (只在主机上跑)
def _load(repo, files, split):
    from datasets import load_dataset
    # 只下需要的分片; 仓库说明里登记的其他切分不在 → 关掉切分校验 (与 eval_medvqa.py 相同)
    return load_dataset(repo, data_files={split: files}, split=split, verification_mode="no_checks")


def _check_count(name, n, allow):
    if n != EXPECTED_TRAIN[name]:
        msg = f"{name} 训练部分 {n} 条, 公开说明是 {EXPECTED_TRAIN[name]} 条"
        if not allow:
            sys.exit(msg + "。先核对再继续 (确认无误可加 --allow-count-mismatch)。")
        print("注意: " + msg)


def _pneumonia_train(allow):
    from datasets import load_dataset
    ds = load_dataset("danjacobellis/pneumoniamnist_224", split="train")
    feats = ds.features
    img_col = next((c for c, f in feats.items() if type(f).__name__ == "Image"), None)
    lab_col = next((c for c in ("label", "labels") if c in feats), None)
    if img_col is None or lab_col is None:
        sys.exit(f"PneumoniaMNIST 镜像的列认不出: {feats}")
    names = getattr(feats[lab_col], "names", None)
    if names and "pneu" not in str(names[1]).lower():
        sys.exit(f"标签 1 不是肺炎: {names}")
    if img_col != "image":
        ds = ds.rename_column(img_col, "image")
    _check_count("pneumonia", len(ds), allow)
    return ds, lab_col


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--image-dir", required=True)
    ap.add_argument("--only", nargs="*", choices=sorted(EXPECTED_TRAIN), default=sorted(EXPECTED_TRAIN))
    ap.add_argument("--vqarad", type=int, default=0, help="0 = 全部")
    ap.add_argument("--pathvqa", type=int, default=5000)
    ap.add_argument("--medqa", type=int, default=3000)
    ap.add_argument("--pneumonia", type=int, default=2000, help="肺炎 / 正常各一半")
    ap.add_argument("--pneumonia-test-npz", default=str(REPO / "data/raw/medmnist/pneumoniamnist_224_test.npz"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--allow-count-mismatch", action="store_true")
    ap.add_argument("--keep-test-image-rows", action="store_true",
                    help="VQA 类保留问测试部分图片的训练题 (默认去掉, 见 build_vqa)")
    args = ap.parse_args()
    out, imgs = Path(args.out_dir), Path(args.image_dir)
    out.mkdir(parents=True, exist_ok=True)
    stats_path = out / "sft6_stats.json"
    all_stats = json.loads(stats_path.read_text(encoding="utf-8")) if stats_path.is_file() else {}

    for name in args.only:
        print(f"==== {name}", flush=True)
        if name in ("vqarad", "pathvqa"):
            repo = {"vqarad": "flaviagiammarino/vqa-rad", "pathvqa": "flaviagiammarino/path-vqa"}[name]
            train, test = _load(repo, "data/train-*.parquet", "train"), _load(repo, "data/test-*.parquet", "test")
            _check_count(name, len(train), args.allow_count_mismatch)
            rows, st = build_vqa(train, test, imgs / name, getattr(args, name), args.seed,
                                 exclude_test_images=not args.keep_test_image_rows)
        elif name == "medqa":
            repo = "GBaker/MedQA-USMLE-4-options"
            train = _load(repo, "phrases_no_exclude_train.jsonl", "train")
            test = _load(repo, "phrases_no_exclude_test.jsonl", "test")
            _check_count(name, len(train), args.allow_count_mismatch)
            rows, st = build_medqa(train, test, args.medqa, args.seed)
        else:
            import numpy as np
            from PIL import Image
            train, lab_col = _pneumonia_train(args.allow_count_mismatch)
            test_images = [Image.fromarray(a) for a in np.load(args.pneumonia_test_npz)["test_images"]]
            rows, st = build_pneumonia(train, test_images, imgs / name, args.pneumonia, args.seed, lab_col)
        st["seed"] = args.seed
        fn = out / f"{name}_sft_train.json"
        fn.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        all_stats[name] = st
        stats_path.write_text(json.dumps(all_stats, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"{fn}: {len(rows)} 条 | {json.dumps(st, ensure_ascii=False)}", flush=True)


if __name__ == "__main__":
    main()
