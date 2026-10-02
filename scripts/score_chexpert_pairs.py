"""用队友 (Topic 1) 的图文对齐模型给 CheXpert Plus 的「胸片 + 报告」配对打分, 供实验 B3 筛选 CPT 数据。

只读使用队友的代码、检查点和数据 (他的原话: 可以读, 不要改); 本脚本在我们自己的 conda 环境 (chunqian) 里运行,
所有输出只写到 /workspace/chunqian/ 下。CheXpert Plus 是受控数据: 输出的分数文件含病人编号, 留在这台机器上,
不进 git、不拷出、不贴到聊天里; 屏幕上只打印汇总数字。

做法与队友训练时一致 (radpair/src/data/dataset.py 与 experiments/<ckpt>/config.yaml):
  图片 = Chambon JPG 缓存 → RGB → Resize 224 → ImageNet 归一化; 文本 = report_text (Findings + Impression),
  Bio_ClinicalBERT 分词, 截到 256 token; 分数 = 图像向量与文本向量的余弦相似度 (两者已 L2 归一化, 即点积)。

第一步务必先跑验证集 (--split val --retrieval): 算 i2t R@1, 应接近队友 metrics.json 里的 0.0191 (1,518 张的图库)。
对上了才说明模型在我们的环境 (peft 版本可能不同) 里加载正确; 加载时若打印 "warn missing keys", 也说明有权重没对上。

用法 (user0 容器):
  source /opt/conda/etc/profile.d/conda.sh && conda activate chunqian
  cd /workspace/chunqian/MedLoRA
  python scripts/score_chexpert_pairs.py --split val --retrieval            # 校验, 几分钟
  python scripts/score_chexpert_pairs.py --pairs <我们自己的新样本清单.csv> --out <...>   # 给新数据打分 (之后)
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import torch

RADPAIR = Path("/workspace/rafi/radpair")
PAIRS = Path("/workspace/rafi/chexpert_plus_subset/pairs_with_text_labels.csv")
CACHE = Path("/workspace/rafi/chexpert_plus_subset/images_chambon")
OUT_DIR = Path("/workspace/chunqian/runs/b3_scores")
KEEP_COLS = ["deid_patient_id", "patient_report_date_order", "path_to_dcm", "our_split"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--radpair", default=str(RADPAIR))
    ap.add_argument("--ckpt", default="experiments/plus_g_peft_001/checkpoint.pt")
    ap.add_argument("--pairs", default=str(PAIRS), help="含 path_to_dcm 与 report_text 的 CSV")
    ap.add_argument("--image-cache-root", default=str(CACHE))
    ap.add_argument("--split", default="val", help="按 our_split 过滤; all = 不过滤")
    ap.add_argument("--retrieval", action="store_true", help="另算图库内检索 R@1/5/10 (校验用)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-text-length", type=int, default=256)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=16, help="CPU 线程数; 开太多反而慢")
    ap.add_argument("--workers", type=int, default=8, help="并行读图的进程数 (队友训练时也是 8); 0 = 主进程读")
    ap.add_argument("--out", default=None, help="分数 CSV 路径, 默认 /workspace/chunqian/runs/b3_scores/<split>_scores.csv")
    args = ap.parse_args()

    sys.path.insert(0, args.radpair)
    from src.api.embed import _resolve_path, embed_image, embed_text, load_dual_encoder  # 队友的代码, 只读导入
    from src.data.dataset import chambon_cache_jpg_path, image_transform
    from transformers import AutoTokenizer
    from PIL import Image

    torch.set_num_threads(args.threads)
    rows = list(csv.DictReader(open(args.pairs, encoding="utf-8")))
    if args.split != "all":
        rows = [r for r in rows if r.get("our_split") == args.split]
    print(f"配对数: {len(rows)} (split={args.split})", flush=True)

    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    model = load_dual_encoder(args.ckpt, device=device)
    ckpt_cfg = torch.load(Path(args.radpair) / args.ckpt if not Path(args.ckpt).is_absolute() else args.ckpt,
                          map_location="cpu", weights_only=False).get("cfg", {})
    text_encoder = _resolve_path(str(ckpt_cfg.get("text_encoder")))  # 与 load_dual_encoder 同一解析方式
    tok = AutoTokenizer.from_pretrained(text_encoder)
    tfm = image_transform()
    cache_root = Path(args.image_cache_root)

    # 读图 + 缩放是瓶颈 (单进程约 2.5 秒一张), 交给 DataLoader 多进程做; 每张图的处理与原来完全相同, 顺序也不变
    class Imgs(torch.utils.data.Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            return tfm(Image.open(chambon_cache_jpg_path(cache_root, rows[i]["path_to_dcm"])).convert("RGB"))

    loader = torch.utils.data.DataLoader(Imgs(), batch_size=args.batch, shuffle=False, num_workers=args.workers)
    img_embs, txt_embs = [], []
    t0 = time.time()
    for b, pix in enumerate(loader):
        s = b * args.batch
        chunk = rows[s:s + len(pix)]
        enc = tok([r["report_text"] for r in chunk], max_length=args.max_text_length, padding="max_length",
                  truncation=True, return_tensors="pt")
        img_embs.append(embed_image(model, pix.to(device)).float().cpu())
        txt_embs.append(embed_text(model, enc["input_ids"].to(device), enc["attention_mask"].to(device)).float().cpu())
        if b % 10 == 0:
            print(f"  {s + len(pix)}/{len(rows)}  {time.time() - t0:.0f}s", flush=True)
    I, T = torch.cat(img_embs), torch.cat(txt_embs)
    scores = (I * T).sum(-1)

    summary = {"n": len(rows), "split": args.split, "ckpt": args.ckpt, "device": str(device),
               "seconds": round(time.time() - t0, 1),
               "score_mean": round(scores.mean().item(), 4), "score_std": round(scores.std().item(), 4),
               "score_min": round(scores.min().item(), 4), "score_max": round(scores.max().item(), 4),
               "score_quartiles": [round(q, 4) for q in torch.quantile(scores, torch.tensor([.25, .5, .75])).tolist()]}
    if args.retrieval:
        sim = I @ T.T
        rank = (sim > sim.diag().unsqueeze(1)).sum(1)          # 图 → 文: 正确报告前面有几份
        rank_t = (sim.T > sim.diag().unsqueeze(1)).sum(1)      # 文 → 图
        for k in (1, 5, 10):
            summary[f"i2t_R@{k}"] = round((rank < k).float().mean().item(), 4)
            summary[f"t2i_R@{k}"] = round((rank_t < k).float().mean().item(), 4)
        summary["chance_R@1"] = round(1 / len(rows), 6)
        summary["mismatched_score_mean"] = round(((sim.sum() - sim.diag().sum()) / (sim.numel() - len(rows))).item(), 4)

    out = Path(args.out) if args.out else OUT_DIR / f"{args.split}_scores.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(KEEP_COLS + ["score"])
        for r, sc in zip(rows, scores.tolist()):
            w.writerow([r.get(c, "") for c in KEEP_COLS] + [round(sc, 6)])
    (out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"分数文件 (含病人编号, 只留在这台机器): {out}")


if __name__ == "__main__":
    main()
