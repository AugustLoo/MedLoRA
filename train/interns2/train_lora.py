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

三步训练 (docs/PLAN_35B_CPT.md) 用到的额外开关:
  --mode pt            纯文字增量预训练: 数据行是 {"text": ...}, 整段文字都算损失 (不套对话模板, 末尾加结束符)
  --init-adapter DIR   从上一步的补丁开始训 (LoRA 设置必须相同); 与 --resume 不同, 优化器和步数从零开始
  --eval-data FILE     验证集: 训练前、每 --eval-steps 步、结束时各算一次平均损失 (按 token 加权), 写进日志 (比较学习率用)

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

    def segments(self, row):
        """多轮纯文字对话 (第②步的华佗 / DISC 数据): 整段套一次模板, 再按顺序在里面找到每一轮回答的位置,
        切成 [(文字, 是否计损失), …]。每一轮回答从内容开头算到它的回合结束符 (含); 用户的话和模板标记都不计损失。
        某一轮回答在模板文本里找不到时返回 None (该条跳过), 不去猜。"""
        msgs = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]} for m in row["messages"]]
        full = self.proc.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
        out, cur = [], 0
        for m in row["messages"]:
            if m["role"] != "assistant":
                continue
            pos = full.find(m["content"], cur)
            if pos < 0:
                return None
            end = full.find(self.end, pos + len(m["content"])) if self.end else -1
            end = end + len(self.end) if end >= 0 else pos + len(m["content"])
            out += [(full[cur:pos], False), (full[pos:end], True)]
            cur = end
        return out

    def _multi_turn(self, row):
        segs = self.segments(row)
        if segs is None:
            return None
        ids, labels = [], []
        for text, train in segs:
            t = self.tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            if train and len(ids) + len(t) > self.cutoff:
                break  # 超长: 只保留放得下的前几轮 (至少要有一轮回答)
            if not train and len(ids) + len(t) >= self.cutoff:
                break
            ids.append(t)
            labels.append(t if train else torch.full_like(t, -100))
        if not any((x != -100).any() for x in labels):
            return None
        ids, labels = torch.cat(ids), torch.cat(labels)
        if (labels[-1] == -100).item():  # 末尾是没有回答跟着的用户话, 去掉
            keep = int((labels != -100).nonzero().max()) + 1
            ids, labels = ids[:keep], labels[:keep]
        return {"input_ids": ids, "labels": labels, "attention_mask": torch.ones_like(ids)}

    def __getitem__(self, i):
        row = self.rows[i]
        if len(row["messages"]) > 2 and not row.get("images"):
            return self._multi_turn(row)
        prompt, target, img = self.render(row)
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


class PretrainData(Dataset):
    """纯文字增量预训练: 每行 {"text": ...} (数据脚本已切成约 1,000 token 一段); 不套对话模板, 末尾加结束符,
    所有 token 都算损失 (与 3B 的 B1 / LLaMA-Factory stage=pt 相同)。超过 cutoff 的段截断, 不丢弃。"""

    def __init__(self, files, processor, cutoff, limit=0, seed=42):
        self.tok, self.cutoff = processor.tokenizer, cutoff
        rows = []
        for f in files:
            part = json.loads(Path(f).read_text(encoding="utf-8"))
            if part and "text" not in part[0]:
                sys.exit(f"{f} 不是预训练格式 (每行要有 \"text\"); 对话格式的数据请用 --mode sft")
            log(f"数据 {f}: {len(part)} 段")
            rows += part
        random.Random(seed).shuffle(rows)
        self.rows = rows[:limit] if limit else rows
        eot = self.tok.convert_tokens_to_ids("<|endoftext|>")
        self.eos_id = eot if eot is not None and eot != self.tok.unk_token_id else self.tok.eos_token_id

    def __len__(self):
        return len(self.rows)

    def render(self, row):
        """与 SFTData.render 同形: (提示词, 计损失的文本, 图片); 预训练没有提示词, 整段都计损失。"""
        return "", row["text"], None

    def __getitem__(self, i):
        ids = self.tok(self.rows[i]["text"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        ids = torch.cat([ids[: self.cutoff - 1], torch.tensor([self.eos_id], dtype=ids.dtype)])
        return {"input_ids": ids, "labels": ids.clone(), "attention_mask": torch.ones_like(ids)}


def adapter_mismatch(saved: dict, rank: int, alpha: int, target: str) -> list[str]:
    """--init-adapter 的补丁必须和本次 LoRA 设置一致, 否则权重形状或挂载位置对不上。返回不一致项 (空 = 一致)。"""
    out = []
    if saved.get("r") != rank:
        out.append(f"rank {saved.get('r')} ≠ {rank}")
    if saved.get("lora_alpha") != alpha:
        out.append(f"alpha {saved.get('lora_alpha')} ≠ {alpha}")
    if saved.get("target_modules") != target:
        out.append(f"target_modules 不同: {saved.get('target_modules')!r}")
    return out


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
    if args.init_adapter:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        src = Path(args.init_adapter)
        bad = adapter_mismatch(json.loads((src / "adapter_config.json").read_text(encoding="utf-8")),
                               args.rank, args.alpha, LORA_TARGETS[args.targets])
        if bad:
            sys.exit(f"--init-adapter {src} 的 LoRA 设置和本次不同: {'; '.join(bad)}")
        weights = load_file(str(src / "adapter_model.safetensors"))
        res = set_peft_model_state_dict(model, weights)
        unexpected = list(getattr(res, "unexpected_keys", []) or [])
        if unexpected:
            sys.exit(f"--init-adapter 有 {len(unexpected)} 个权重对不上, 例如 {unexpected[:3]}")
        log(f"从 {src} 载入上一步的补丁: {len(weights)} 个张量")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    n_mod = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    log(f"LoRA 挂在 {n_mod} 个线性层上; 可训练 {trainable:,} / {total:,} = {100 * trainable / total:.3f}%")
    return model, processor


def first_device(model):
    return next(p.device for p in model.parameters())


def peak_mem():
    return {i: round(torch.cuda.max_memory_allocated(i) / 2**30, 1) for i in range(torch.cuda.device_count())}


@torch.no_grad()
def evaluate(model, loader, dev0, max_batches=0):
    """验证集平均损失, 按计损失的 token 数加权 (模型返回的是每个 batch 内的平均)。"""
    model.eval()
    tot, n_tok = 0.0, 0
    for b, batch in enumerate(loader):
        if max_batches and b >= max_batches:
            break
        if batch is None:
            continue
        batch = {k: v.to(dev0) for k, v in batch.items() if k in MODEL_KEYS + ("labels",)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        k = int((batch["labels"][:, 1:] != -100).sum())
        tot += loss.item() * k
        n_tok += k
    model.train()
    return tot / max(1, n_tok), n_tok


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
    ap.add_argument("--mode", choices=["sft", "pt"], default="sft",
                    help="sft = 对话格式只算答案 (默认); pt = 纯文字增量预训练, 整段都算损失")
    ap.add_argument("--init-adapter", help="从这个补丁目录开始训 (上一步的产物; LoRA 设置须相同)")
    ap.add_argument("--eval-data", nargs="+", default=[], help="验证集文件 (格式与 --mode 一致)")
    ap.add_argument("--eval-steps", type=int, default=100, help="每多少步算一次验证损失")
    ap.add_argument("--eval-limit", type=int, default=0, help="验证集只用前 N 条 (0 = 全部)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / "run_args.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    path_map = [tuple(x.split("=", 1)) for x in args.path_map]

    model, processor = load(args)
    limit = args.limit or (args.batch * args.accum * args.smoke_steps if args.smoke_steps else 0)

    def make_data(files, lim):
        if args.mode == "pt":
            return PretrainData(files, processor, args.cutoff, limit=lim, seed=args.seed)
        return SFTData(files, processor, args.cutoff, args.max_pixels, path_map, limit=lim, seed=args.seed)

    data = make_data(args.data, limit)

    # 看一眼前两条样本: sft 只有答案部分参与损失; pt 整段参与 (只打印开头和结尾)
    for k in range(min(2, len(data))):
        prompt, target, img = data.render(data.rows[k])
        if args.mode == "sft":
            shown = prompt.replace("<|image_pad|>", "")
            kind = "带图" if img is not None else "纯文字"
            log(f"样本 {k} 提示词 ({kind}, 图片占位已省略):\n" + shown[-600:])
        ex = data[k]
        if ex is not None:
            tgt = ex["labels"][ex["labels"] != -100]
            text = processor.tokenizer.decode(tgt)
            shown = text if args.mode == "sft" else text[:200] + " …… " + text[-120:]
            log(f"样本 {k}: 总长 {len(ex['input_ids'])} token, 其中计损失 {len(tgt)} 个 -> {shown!r}")

    pad_id = processor.tokenizer.pad_token_id if processor.tokenizer.pad_token_id is not None else processor.tokenizer.eos_token_id
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(data, batch_size=args.batch, shuffle=True, generator=gen, num_workers=args.workers,
                        collate_fn=make_collate(pad_id), persistent_workers=args.workers > 0)
    eval_loader = None
    if args.eval_data:
        eval_set = make_data(args.eval_data, args.eval_limit)
        eval_loader = DataLoader(eval_set, batch_size=args.batch, shuffle=False, num_workers=min(2, args.workers),
                                 collate_fn=make_collate(pad_id))
        log(f"验证集 {len(eval_set)} 条, 每 {args.eval_steps} 步算一次")
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

    def run_eval(at_step):
        if eval_loader is None:
            return
        t0 = time.time()
        loss, n_tok = evaluate(model, eval_loader, dev0, max_batches=2 if args.smoke_steps else 0)
        rec = {"type": "eval", "step": at_step, "eval_loss": round(loss, 4), "eval_tokens": n_tok,
               "eval_sec": round(time.time() - t0, 1)}
        logf.write(json.dumps(rec, ensure_ascii=False) + "\n")
        logf.flush()
        log(json.dumps(rec, ensure_ascii=False))

    if step == 0:
        run_eval(0)
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
            if args.eval_steps and step % args.eval_steps == 0 and step != total_steps and not args.smoke_steps:
                run_eval(step)
            if step % args.save_steps == 0 and not args.smoke_steps:
                ckpt.mkdir(exist_ok=True)
                model.save_pretrained(ckpt)
                torch.save({"opt": opt.state_dict(), "sched": sched.state_dict(), "step": step}, ckpt / "state.pt")
                log(f"检查点已存: 第 {step} 步")
            if step >= total_steps:
                done = True
                break
        epoch += 1

    run_eval(step)
    if args.smoke_steps:
        log(f"冒烟结束。各卡显存峰值 (GiB): {peak_mem()}")
        log("没有保存 adapter。确认显存和每步耗时后, 去掉 --smoke-steps 正式训练。")
        return
    model.save_pretrained(out)
    log(f"训练完成, adapter 已保存到 {out}; 共 {step} 步, 用时 {(time.time() - t_start) / 3600:.2f} 小时, 跳过超长样本 {skipped} 条")


if __name__ == "__main__":
    main()
