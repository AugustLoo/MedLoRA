"""Intern-S2-Preview (35B MoE) 的 LoRA SFT: 在 3B 实验的同一份数据上重复「回放 0 / 回放 300」。

为什么不用 LLaMA-Factory: 这个模型是自定义代码 (trust_remote_code), 结构为 Qwen3.5-MoE (线性注意力 +
每 4 层一层全注意力, 256 个路由专家打包成一个 3D 参数) + 视觉塔 + 时间序列模块, LLaMA-Factory 不认识它。
这里直接用 transformers + peft, 训练循环自己写, 其余设置与 3B 对齐:

    与 3B 相同: 数据文件 (data/processed/*.json, 同一份 SLAKE 与回放样本)、lr 1e-4、3 轮、有效 batch 16、
                cosine + 5% warmup、截断 1024、图片 ≤ 262144 像素、LoRA r16 / alpha 32 / dropout 0.05、seed 42、
                视觉塔冻结、梯度裁剪 1.0
    与 3B 不同: LoRA 只挂语言模型的注意力 (全注意力 q/k/v/o + 线性注意力 in_proj_qkv / in_proj_z / out_proj)
                和共享专家 (gate/up/down)。256 个路由专家是打包的 3D 参数, 不是 nn.Linear, 不挂;
                路由器、词表、时间序列模块全部冻结。关闭思考模式 (与评估一致)。

模型按层切到多张卡上 (device_map="auto", 单进程, 同一时刻只有一张卡在算), 69 GB 权重 4 张 32 GB 卡放得下。

用法 (在 ubuntu 主机上, 先挑空闲卡):
  # 冒烟: 只跑 3 个优化步, 打印可训练参数、标签掩码、每步耗时和各卡显存峰值
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train/interns2/train_lora.py \
      --data data/processed/slake_train.json data/processed/pubmedqa_sft_train_300.json \
      --output outputs/interns2_smoke --smoke-steps 3
  # 正式 (回放 300)
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train/interns2/train_lora.py \
      --data data/processed/slake_train.json data/processed/pubmedqa_sft_train_300.json \
      --output outputs/interns2_mix_300
  # 断了从最近的检查点接着跑: 同一条命令加 --resume

日志: <output>/train_log.jsonl (每 10 步一行, 含 loss / lr / 每步秒数 / 预计剩余时间); 结束时保存 LoRA adapter。
"""
from __future__ import annotations

import os

# 预处理线程: 学院服务器上 128 线程比 1 线程慢 374 倍 (docs/SERVER.md), 必须在 import torch 之前设
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

MODEL_DEFAULT = "/home/ubuntu/Large-Model-Service-Interns2/models/Intern-S2-Preview"
# LoRA 目标: 只在语言模型里 (不碰 model.visual / model.time_series)
_ATTN = (r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"
         r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)")
_SHARED = r"|mlp\.shared_expert\.(gate_proj|up_proj|down_proj)"
LORA_TARGETS = {
    "attn_shared": r"model\.language_model\.layers\.\d+\.(" + _ATTN + _SHARED + ")",  # 默认, 主实验用的
    "attn": r"model\.language_model\.layers\.\d+\.(" + _ATTN + ")",                 # 消融: 不挂共享专家
}
MODEL_KEYS = ("input_ids", "attention_mask", "pixel_values", "image_grid_thw")


def log(msg: str):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def resize_to(img: Image.Image, max_pixels: int) -> Image.Image:
    img = img.convert("RGB")
    w, h = img.size
    if w * h > max_pixels:
        s = math.sqrt(max_pixels / (w * h))
        img = img.resize((max(28, int(w * s)), max(28, int(h * s))), Image.LANCZOS)
    return img


# ---------------------------------------------------------------- 数据
class SFTData(Dataset):
    """读 LLaMA-Factory 的 sharegpt 文件 (与 3B 同一份), 在线编码成 模型输入 + 只在答案上算损失的 labels。"""

    def __init__(self, files, processor, cutoff, max_pixels, path_map, limit=0, seed=42):
        self.proc, self.tok = processor, processor.tokenizer
        self.cutoff, self.max_pixels, self.path_map = cutoff, max_pixels, path_map
        rows = []
        for f in files:
            part = json.loads(Path(f).read_text(encoding="utf-8"))
            log(f"数据 {f}: {len(part)} 条")
            rows += part
        random.Random(seed).shuffle(rows)
        self.rows = rows[:limit] if limit else rows
        # 回合结束符: Qwen 系模板是 <|im_end|>; 词表里没有才退回 tokenizer 的 eos
        im_end = self.tok.convert_tokens_to_ids("<|im_end|>")
        if im_end is not None and im_end != self.tok.unk_token_id:
            self.end = "<|im_end|>"
        else:
            self.end = self.tok.eos_token

    def __len__(self):
        return len(self.rows)

    def _map(self, p: str) -> str:
        for old, new in self.path_map:
            if p.startswith(old):
                return new + p[len(old):]
        return p

    def render(self, row):
        """返回 (prompt 文本, 答案目标文本, 图片或 None)。"""
        user, answer = row["messages"][0]["content"], row["messages"][1]["content"]
        img = None
        if row.get("images"):
            img = resize_to(Image.open(self._map(row["images"][0])), self.max_pixels)
            content = [{"type": "image"}, {"type": "text", "text": user.replace("<image>", "", 1)}]
        else:
            content = [{"type": "text", "text": user}]
        msgs = [{"role": "user", "content": content}]
        prompt = self.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        full = self.proc.apply_chat_template(msgs + [{"role": "assistant", "content": [{"type": "text", "text": answer}]}],
                                             tokenize=False, enable_thinking=False)
        if full.startswith(prompt) and answer in full[len(prompt):]:
            target = full[len(prompt):]
            if self.end and self.end in target:  # 截到回合结束符为止 (含), 之后的换行不学
                target = target[: target.index(self.end) + len(self.end)]
        else:  # 模板对历史回合的写法与生成提示不一致时, 退回「答案 + 结束符」
            target = answer + (self.end or "")
        return prompt, target, img

    def __getitem__(self, i):
        prompt, target, img = self.render(self.rows[i])
        enc = self.proc(text=[prompt], images=[img] if img is not None else None, return_tensors="pt")
        tgt = self.tok(target, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([enc["input_ids"][0], tgt])
        if len(ids) > self.cutoff:
            return None  # 图片 token 不能截断, 超长样本整条跳过并计数
        labels = torch.cat([torch.full((enc["input_ids"].shape[1],), -100), tgt])
        item = {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}
        if img is not None:
            item["pixel_values"] = enc["pixel_values"]
            item["image_grid_thw"] = enc["image_grid_thw"]
        return item


def make_collate(pad_id: int):
    def collate(items):
        items = [x for x in items if x is not None]
        if not items:
            return None
        n = max(len(x["input_ids"]) for x in items)
        def pad(key, val):
            return torch.stack([torch.cat([x[key], torch.full((n - len(x[key]),), val, dtype=x[key].dtype)]) for x in items])
        batch = {"input_ids": pad("input_ids", pad_id), "labels": pad("labels", -100),
                 "attention_mask": pad("attention_mask", 0)}
        imgs = [x for x in items if "pixel_values" in x]
        if imgs:
            batch["pixel_values"] = torch.cat([x["pixel_values"] for x in imgs])
            batch["image_grid_thw"] = torch.cat([x["image_grid_thw"] for x in imgs])
        return batch
    return collate


# ---------------------------------------------------------------- 模型
def load(args):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    n_gpu = torch.cuda.device_count()
    log(f"可见 GPU {n_gpu} 张 (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '未设置')}), 每张上限 {args.max_memory}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, trust_remote_code=True, dtype=torch.bfloat16, device_map="auto",
        max_memory={i: args.max_memory for i in range(n_gpu)})  # 不指定: 各子模块自选 (时间序列模块不支持 sdpa, 指定会报错)
    model.config.use_cache = False
    for p in model.parameters():
        p.requires_grad_(False)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    cfg = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                     target_modules=LORA_TARGETS[args.targets], bias="none")
    model = get_peft_model(model, cfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    n_mod = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    log(f"LoRA 挂在 {n_mod} 个线性层上; 可训练 {trainable:,} / {total:,} = {100 * trainable / total:.3f}%")
    return model, processor


def first_device(model):
    return next(p.device for p in model.parameters())


def peak_mem():
    return {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())}


# ---------------------------------------------------------------- 训练
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=2, help="每次前向的样本数")
    ap.add_argument("--accum", type=int, default=8, help="梯度累积; batch × accum = 有效 batch, 与 3B 同为 16")
    ap.add_argument("--warmup", type=float, default=0.05)
    ap.add_argument("--cutoff", type=int, default=1024)
    ap.add_argument("--max-pixels", type=int, default=262144)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--targets", choices=sorted(LORA_TARGETS), default="attn_shared",
                    help="LoRA 挂在哪些层; attn = 只挂注意力 (消融)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-memory", default="19GiB", help="每张卡放权重的上限, 其余留给激活")
    ap.add_argument("--path-map", action="append", default=[],
                    help="图片路径前缀替换 OLD=NEW, 数据文件里是容器路径时用; 可重复")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 条样本 (调试)")
    ap.add_argument("--smoke-steps", type=int, default=0, help="只跑 N 个优化步就停, 打印显存与速度")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    path_map = [tuple(x.split("=", 1)) for x in args.path_map]

    model, processor = load(args)
    data = SFTData(args.data, processor, args.cutoff, args.max_pixels, path_map,
                   limit=args.limit or (args.batch * args.accum * args.smoke_steps if args.smoke_steps else 0),
                   seed=args.seed)

    # 看一眼第一条样本: 只有答案部分参与损失
    for k in range(min(2, len(data))):
        prompt, target, img = data.render(data.rows[k])
        shown = prompt.replace("<|image_pad|>", "")
        kind = "带图" if img is not None else "纯文字"
        log(f"样本 {k} 提示词 ({kind}, 图片占位已省略):\n" + shown[-600:])
        ex = data[k]
        if ex is not None:
            tgt = ex["labels"][ex["labels"] != -100]
            log(f"样本 {k}: 总长 {len(ex['input_ids'])} token, 其中计损失 {len(tgt)} 个 -> {processor.tokenizer.decode(tgt)!r}")

    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, generator=gen, num_workers=args.workers,
                        collate_fn=make_collate(pad_id), persistent_workers=args.workers > 0)
    steps_per_epoch = len(loader) // args.accum
    total_steps = args.smoke_steps or max(1, int(steps_per_epoch * args.epochs))
    warm = int(total_steps * args.warmup)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / max(1, warm) if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    log(f"{len(data)} 条样本, batch {args.batch} × 累积 {args.accum} = 有效 {args.batch * args.accum}; "
        f"每轮 {steps_per_epoch} 步, 共 {total_steps} 步")

    step, skip_batches = 0, 0
    ckpt = out / "checkpoint"
    if args.resume and (ckpt / "state.pt").exists():
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file
        set_peft_model_state_dict(model, load_file(str(ckpt / "adapter_model.safetensors")))
        st = torch.load(ckpt / "state.pt", map_location="cpu", weights_only=False)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step = st["step"]
        skip_batches = step * args.accum
        log(f"从检查点恢复: 第 {step} 步, 跳过已训练的 {skip_batches} 个 batch")

    dev0 = first_device(model)
    model.train()
    logf = open(out / "train_log.jsonl", "a", encoding="utf-8")
    t_start, t_last, skipped, running = time.time(), time.time(), 0, []
    last_logged = step
    micro, done = 0, False
    epoch = 0
    while not done:
        gen.manual_seed(args.seed + epoch)  # 每轮顺序固定, 恢复时能跳到同一位置
        for batch in loader:
            if skip_batches:
                skip_batches -= 1
                micro += 1
                continue
            if batch is None:
                skipped += args.batch
                continue
            batch = {k: v.to(dev0) for k, v in batch.items() if k in MODEL_KEYS + ("labels",)}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss / args.accum
            loss.backward()
            running.append(loss.item() * args.accum)
            micro += 1
            if micro % args.accum:
                continue
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            now = time.time()
            if step % 10 == 0 or args.smoke_steps or step == total_steps:
                sec = (now - t_last) / max(1, step - last_logged)
                rec = {"step": step, "total": total_steps, "epoch": round(step / max(1, steps_per_epoch), 3),
                       "loss": round(sum(running) / len(running), 4), "lr": sched.get_last_lr()[0],
                       "sec_per_step": round(sec, 2), "elapsed_min": round((now - t_start) / 60, 1),
                       "eta_min": round(sec * (total_steps - step) / 60, 1), "skipped_long": skipped}
                if args.smoke_steps:
                    rec["peak_mem_gib"] = peak_mem()
                logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
                logf.flush()
                log(json.dumps(rec, ensure_ascii=False))
                running, t_last, last_logged = [], now, step
            if step % args.save_steps == 0 and not args.smoke_steps:
                ckpt.mkdir(exist_ok=True)
                model.save_pretrained(ckpt)
                torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, ckpt / "state.pt")
                log(f"检查点已存: 第 {step} 步")
            if step >= total_steps:
                done = True
                break
        epoch += 1

    if args.smoke_steps:
        log(f"冒烟结束。各卡显存峰值 (GiB): {peak_mem()}")
        log("没有保存 adapter。确认显存和每步耗时后, 去掉 --smoke-steps 正式训练。")
        return
    model.save_pretrained(out)
    log(f"训练完成, adapter 已保存到 {out}; 共 {step} 步, 用时 {(time.time() - t_start) / 3600:.2f} 小时, 跳过超长样本 {skipped} 条")


if __name__ == "__main__":
    main()
