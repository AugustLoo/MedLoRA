"""S2-6datasets 的四个新训练来源 (只用训练部分): VQA-RAD / PathVQA / MedQA / PneumoniaMNIST。

设计: docs/superpowers/specs/2026-10-08-s2-6datasets-design.md
提示词与评测逐字相同 (medvlm.prompts; PneumoniaMNIST 与 eval/eval_pneumonia.py 的是非题同一句),
输出与 slake_train.json 同格式 (messages + images), train/interns2/train_lora.py 直接读。

防泄漏 (自动, 统计写进 <out-dir>/sft6_stats.json):
  - 训练样本与测试部分「同图同题」(VQA) / 「同题干」(MedQA) / 「同图」(PneumoniaMNIST) 的剔除;
  - 训练部分内部完全重复的样本只留一条;
  - 统计训练 / 测试共用的图片数 (VQA-RAD 已知有), 报告里注明。
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
