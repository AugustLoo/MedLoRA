"""把 LoRA adapter 合并进 Intern-S2-Preview 的权重, 存成一份完整模型, 供 LMDeploy 直接加载评估。

为什么合并: LoRA 挂在线性注意力的 in_proj_qkv / in_proj_z 上, LMDeploy 推理时可能把这些投影融合成别的形状,
直接加载 adapter 没把握。合并后就是一个普通的完整模型目录, 用和基座完全相同的方式起服务、走接口评估。

做法:
  1. 与训练相同的方式把基座按层切到多张卡上 (device_map="auto")
  2. 加载 adapter, merge_and_unload, save_pretrained (bf16, 5 GB 一片)
  3. 把基座目录里的其余文件 (代码、分词器、处理器、聊天模板、config) 原样拷过来
  4. 核对权重名: 基座里有、合并后没有的 (例如 HF 模型不加载的 MTP 层), 从基座分片里原样补进一个额外分片并写进索引

用法 (ubuntu 主机, 0-3 号卡空闲时):
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train/interns2/merge_lora.py \
      --adapter outputs/interns2_mix_300 --out /home/ubuntu/chunqian/merged/interns2_mix_300
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import torch

MODEL_DEFAULT = "/home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview"


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=MODEL_DEFAULT)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-memory", default="22GiB")
    args = ap.parse_args()

    from peft import PeftModel
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers import AutoModelForImageTextToText

    base, out = Path(args.base), Path(args.out)
    if out.exists() and any(out.glob("*.safetensors")):
        raise SystemExit(f"{out} 里已经有权重了, 不覆盖。确认不要了再手动删。")
    out.mkdir(parents=True, exist_ok=True)

    n = torch.cuda.device_count()
    log(f"加载基座到 {n} 张卡 ...")
    model = AutoModelForImageTextToText.from_pretrained(
        str(base), trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
        max_memory={i: args.max_memory for i in range(n)})
    log(f"加载 adapter {args.adapter} 并合并 ...")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    log(f"保存到 {out} ...")
    model.save_pretrained(str(out), safe_serialization=True, max_shard_size="5GB")
    del model
    torch.cuda.empty_cache()

    # 其余文件原样拷贝 (config.json 也用基座的, 保证与同学部署的模型完全一致)
    for f in base.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)
    for d in base.iterdir():
        if d.is_dir() and not (out / d.name).exists():
            shutil.copytree(d, out / d.name)

    # 核对权重名, 补齐 HF 模型没加载的张量
    bidx = json.loads((base / "model.safetensors.index.json").read_text())
    oidx_fn = out / "model.safetensors.index.json"
    oidx = json.loads(oidx_fn.read_text())
    bkeys, okeys = set(bidx["weight_map"]), set(oidx["weight_map"])
    missing, extra = sorted(bkeys - okeys), sorted(okeys - bkeys)
    log(f"权重名: 基座 {len(bkeys)} 个, 合并后 {len(okeys)} 个; 缺 {len(missing)} 个, 多 {len(extra)} 个")
    if extra:
        log(f"多出来的前 10 个 (名字对不上, 需要检查): {extra[:10]}")
    if missing:
        log(f"缺的前 10 个 (从基座原样补): {missing[:10]}")
        tensors = {}
        by_file = {}
        for k in missing:
            by_file.setdefault(bidx["weight_map"][k], []).append(k)
        for fn, keys in by_file.items():
            with safe_open(str(base / fn), framework="pt") as f:
                for k in keys:
                    tensors[k] = f.get_tensor(k)
        extra_fn = "model-extra-from-base.safetensors"
        save_file(tensors, str(out / extra_fn), metadata={"format": "pt"})
        for k in missing:
            oidx["weight_map"][k] = extra_fn
        oidx_fn.write_text(json.dumps(oidx, indent=2))
        log(f"已补 {len(missing)} 个张量到 {extra_fn}")
    size = sum(f.stat().st_size for f in out.glob("*.safetensors")) / 1e9
    log(f"完成: {out} 权重共 {size:.1f} GB (基座 73.2 GB)")


if __name__ == "__main__":
    main()
