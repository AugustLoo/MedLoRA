"""SLAKE 数据读取。

官方 SLAKE 发布格式 (https://github.com/SuperJohnZhang/Slake):
    <root>/train.json  validate.json  test.json  imgs/xmlab*/source.jpg
每条记录字段: img_id, img_name, question, answer, q_lang(en/zh), location, modality,
             answer_type(OPEN/CLOSED), base_type, content_type, triple, qid

data/download_slake.py 会把 HF 镜像整包拉到 data/raw/SLAKE 并解压。
这里同时兼容: (1) 官方 json+imgs 布局  (2) 已经带 image 列的 HF datasets 格式。
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw" / "SLAKE"

SPLIT_FILES = {"train": "train.json", "validation": "validate.json", "test": "test.json"}


def _find(root: Path, name: str) -> Path | None:
    hits = list(root.rglob(name))
    return hits[0] if hits else None


def load_split(split: str, lang: str = "en", root: Path = RAW) -> list[dict]:
    """返回 list[dict], 每条含 question/answer/answer_type/image_path 或 image。"""
    fn = _find(root, SPLIT_FILES[split])
    if fn is not None:
        img_root = _find(root, "imgs")
        if img_root is None:
            raise FileNotFoundError(f"找到 {fn} 但没找到 imgs/ 目录, 请检查 {root}")
        with open(fn, encoding="utf-8") as f:
            rows = json.load(f)
        out = []
        for r in rows:
            if lang and str(r.get("q_lang", "en")).lower() != lang:
                continue
            out.append({
                "qid": r.get("qid"),
                "question": r["question"],
                "answer": str(r["answer"]),
                "answer_type": r.get("answer_type", "OPEN"),
                "modality": r.get("modality"),
                "content_type": r.get("content_type"),
                "image_path": str(img_root / r["img_name"]),
            })
        return out

    # 退路: HF datasets 格式
    from datasets import load_dataset
    ds = load_dataset("BoKelvin/SLAKE", split=split)
    cols = set(ds.column_names)
    print(f"[slake] HF 列名: {sorted(cols)}")
    out = []
    for r in ds:
        if lang and "q_lang" in cols and str(r["q_lang"]).lower() != lang:
            continue
        out.append({
            "qid": r.get("qid"),
            "question": r["question"],
            "answer": str(r["answer"]),
            "answer_type": r.get("answer_type", "OPEN"),
            "modality": r.get("modality"),
            "content_type": r.get("content_type"),
            "image": r.get("image"),
            "image_path": None,
        })
    return out


def open_image(row: dict) -> Image.Image:
    if row.get("image") is not None:
        return row["image"]
    return Image.open(row["image_path"])
