"""从 configs/*.yaml (Kaggle T4, 4bit + fp16) 派生 configs/bf16/*.yaml (24-32 GB 单卡, 不量化 + bf16)。

改动: 去掉 quantization_bit / quantization_method; fp16 -> bf16; batch 4 x 累积 4 (有效 batch 仍为 16);
adapter_name_or_path 指向同名 output_dir, 所以 B1/B2 的两段式在 bf16 下同样能接上。
用法: python scripts/make_bf16_configs.py
"""
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "configs"
DST = SRC / "bf16"
DST.mkdir(exist_ok=True)

for fn in sorted(SRC.glob("*.yaml")):
    lines = fn.read_text(encoding="utf-8").splitlines()
    out = ["# 由 scripts/make_bf16_configs.py 从 ../%s 派生: 不量化, bf16, batch 4 x 4 (24-32 GB 单卡)" % fn.name]
    for l in lines:
        if re.match(r"^\s*quantization_(bit|method)\s*:", l):
            continue
        l = re.sub(r"^(\s*)fp16:\s*true.*$", r"\1bf16: true", l)
        l = re.sub(r"^(\s*per_device_train_batch_size:)\s*\d+.*$", r"\1 4", l)
        l = re.sub(r"^(\s*per_device_eval_batch_size:)\s*\d+.*$", r"\1 4", l)
        l = re.sub(r"^(\s*gradient_accumulation_steps:)\s*\d+.*$", r"\1 4   # 有效 batch 16", l)
        # 预处理: 8 个 worker + 每批 64 条。默认一批 1000 条时进度条要等一整批才跳一次,
        # 8 个 worker 各分 ~615 条就永远显示 0%, 看着像挂死其实在干活 (2026-09-19 误杀过两次)。
        # 每张图要几秒 CPU, 4919 条单进程需 7-8 小时, 8 进程约 1 小时。两个参数都不影响训出的模型。
        l = re.sub(r"^(\s*preprocessing_num_workers:)\s*\d+.*$", r"\1 8", l)
        out.append(l)
        if l.startswith("preprocessing_num_workers:"):
            out.append("preprocessing_batch_size: 64   # 进度条每 64 条跳一次, 否则一批 1000 条前一直显示 0%")
    (DST / fn.name).write_text("\n".join(out) + "\n", encoding="utf-8")
    print("->", DST / fn.name)
